from functools import lru_cache
import math
from transformers import EncodecModel, AutoProcessor
from encodec.utils import convert_audio
import torchaudio
import torch
from typing import List, Tuple

from gigmate.utils.constants import get_params
from gigmate.utils.device import Device
from gigmate.utils.sequence_utils import get_end_of_sequence_token, get_start_of_sequence_token

AUDIO_CHUNKS_DURATION = 20


@lru_cache(maxsize=1)
def get_codec():
    model = EncodecModel.from_pretrained("facebook/encodec_32khz", normalize=False)
    #print(model.config)
    return model.eval()


@lru_cache(maxsize=1)
def get_processor():
    return AutoProcessor.from_pretrained("facebook/encodec_32khz")


def encode_file(audio_path: str, device: Device, add_start_and_end_tokens: bool = False) -> Tuple[List[torch.Tensor], float]:
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)
    return encode(wav, sr, device, add_start_and_end_tokens=add_start_and_end_tokens)


def get_chunk_length(samples_per_chunk: int, index: int, total_chunks: int, samples_per_token: int, add_start_and_end_tokens: bool) -> int:
    if add_start_and_end_tokens:
        if index == 0:
            if total_chunks == 1:
                return samples_per_chunk - 2 * samples_per_token
            else:
                return samples_per_chunk - 1 * samples_per_token

        if index == total_chunks - 1:
            return samples_per_chunk - 1 * samples_per_token

    return samples_per_chunk


def get_total_chunks(samples_per_chunk: int, num_samples: int, samples_per_token: int, add_start_and_end_tokens: bool) -> int:
    if add_start_and_end_tokens == True:
        return math.ceil((num_samples + 2 * samples_per_token) / samples_per_chunk)
    return math.ceil(num_samples / samples_per_chunk)


def bundle_chunks_and_add_special_tokens(chunks: List[torch.Tensor], encoded_tokens_per_chunk: int, add_start_and_end_token: bool, device: Device) -> List[torch.Tensor]:
    max_seq_len = get_params()['max_seq_len']
    chunks_per_set: int = math.ceil(max_seq_len / encoded_tokens_per_chunk)
    total_chunks = len(chunks)
    total_sets = math.ceil(total_chunks / chunks_per_set)
    chunks_sets = [chunks[i:i + chunks_per_set] for i in range(0, total_chunks, chunks_per_set)]

    final_chunks = []
    
    for i, chunks_set in enumerate(chunks_sets):
        sequence = torch.cat(chunks_set, dim=-1)

        if add_start_and_end_token:
            if i == 0:
                start_of_sequence_token = get_start_of_sequence_token(sequence.shape[-2]).to(device)
                sequence = torch.cat([start_of_sequence_token, sequence], dim=-1)
            if i == total_sets - 1:
                end_of_sequence_token = get_end_of_sequence_token(sequence.shape[-2]).to(device)
                sequence = torch.cat([sequence, end_of_sequence_token], dim=-1)

        final_chunks.append(sequence)

    return final_chunks


def encode(audio: torch.Tensor, sr: int, device: Device, add_start_and_end_tokens: bool = False) -> Tuple[List[torch.Tensor], float]:

    processor = get_processor()
    codec = get_codec()

    wav = convert_audio(audio, sr, processor.sampling_rate, codec.config.audio_channels)
    # split wav in chunks which length will give encoded chunks of max_seq_len length:
    num_samples = wav.shape[1]
    encoded_tokens_per_chunk = 1024 # large values requires a large amount of memory and can cause OOM errors
    samples_per_token = math.ceil(processor.sampling_rate / codec.config.frame_rate)
    samples_per_chunk = math.ceil((encoded_tokens_per_chunk / codec.config.frame_rate) * processor.sampling_rate)
    total_chunks = get_total_chunks(samples_per_chunk, num_samples, samples_per_token, add_start_and_end_tokens)
    chunks = []
    start_index = 0

    # create audio chunks
    for i in range(total_chunks):
        end_index = start_index + get_chunk_length(samples_per_chunk, i, total_chunks, samples_per_token, add_start_and_end_tokens)
        chunk = wav[:, start_index:end_index]
        chunks.append(chunk)
        start_index = end_index
    
    encoded_chunks = []

    # create encoded chunks from audio chunks
    for i, chunk in enumerate(chunks):
        inputs = processor(raw_audio=chunk[0], sampling_rate=processor.sampling_rate, return_tensors="pt")
        bandwidth = 2.2
        codec = codec.to(device)
        result = codec.encode(inputs["input_values"].to(device), inputs["padding_mask"].to(device), bandwidth=bandwidth)
        assert result.audio_codes.shape[0] == 1, 'Multiple chunks returned by codec encoding, expected one'        
        sequence = result.audio_codes[0]
        encoded_chunks.append(sequence)

    encoded_chunks_bundles = bundle_chunks_and_add_special_tokens(encoded_chunks, encoded_tokens_per_chunk, add_start_and_end_tokens, device=device)

    return encoded_chunks_bundles, codec.config.frame_rate


def decode(codes: torch.Tensor, device: Device) -> torch.Tensor:

    codec = get_codec().to(device)
    decoded_wav = codec.decode(codes.unsqueeze(0).to(device), [None])
    output_tensor = decoded_wav['audio_values'].squeeze(0).detach().cpu()
    return output_tensor