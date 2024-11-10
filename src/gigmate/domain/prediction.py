import math
from typing import List, Optional, Tuple, cast
from pydub import AudioSegment
import torchaudio
from gigmate.dataset.dataset import SequenceLengths
from gigmate.domain.sampling import sample_from_logits
from gigmate.model.codec import decode, encode
from encodec.utils import save_audio
import torch
from tqdm import tqdm
import os

from gigmate.utils.audio_utils import generate_random_filename
from gigmate.utils.constants import get_pad_token_id, get_params, get_special_tokens
from gigmate.utils.device import Device
from gigmate.utils.sequence_utils import apply_interleaving, cut_sequence, get_start_of_sequence_token, pad_sequence, remove_special_tokens, revert_interleaving, shift_sequence

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS = 10


def predict_next_token(
        model: torch.nn.Module,
        input: torch.Tensor,
        full_track_sequence: torch.Tensor,
        sequence_lengths: SequenceLengths,
        current_token_index: int,
        incremental: bool,
        temperature: float = DEFAULT_TEMPERATURE,
        use_cache: bool = True,
        cache: Optional[List[torch.Tensor]] = None,
        cache_index: Optional[int] = None,
        encoder_cache: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], torch.Tensor]:

    with torch.inference_mode():
        print(f"Predicting next token at index {current_token_index}...")
        print(f"Current input is {input.shape} {input}")
        print(f'sequence lengths: {sequence_lengths}')
        outputs, updated_cache, updated_encoder_cache = model(
            input,
            conditioning_input=full_track_sequence,
            sequence_lengths=sequence_lengths,
            use_cache=use_cache,
            cache=cache,
            cache_index=cache_index if incremental else None,
            encoder_cache=encoder_cache,
        )
        outputs = outputs.squeeze(0).transpose(0, 1)  # switch codebooks and sequence dimensions
        outputs = outputs[-1 if use_cache and incremental else current_token_index]  # remove batch dimension and take only next token logits
        predicted_tokens = sample_from_logits(outputs, temperature).unsqueeze(0).unsqueeze(2)  # sample and remove last dimension
        print(f"Predicted token is {predicted_tokens}")
    return predicted_tokens.detach().to('cpu'), updated_cache, updated_encoder_cache


def update_next_sequence(previous_next_sequence: torch.Tensor, current_token: torch.Tensor, max_seq_len: int, current_token_index: int):
    """
    Updates the next sequence with the current token, taking into account interleaving and maximum sequence length.

    Returns:
        torch.Tensor: the updated next sequence ready for the next inference iteration
    """
    current_token = current_token.to(previous_next_sequence.device)
    token_sequence_index = current_token_index + 1
    if token_sequence_index >= max_seq_len:
        next_sequence = shift_sequence(previous_next_sequence)
    else:
        next_sequence = previous_next_sequence

    next_sequence[:, :, token_sequence_index] = current_token.reshape((1, 4))
    
    return next_sequence


def complete_sequence(
    model: torch.nn.Module,
    device: Device,
    input_sequence: torch.Tensor,
    frame_rate: float,
    padding_value: int,
    max_output_length_in_seconds: float = DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
    show_progress: bool = False,
    use_cache: bool = True,
) -> torch.Tensor:

    def process_next_note_sequence(
            model,
            next_sequence: torch.Tensor,
            full_track_sequence: torch.Tensor,
            sequence_lengths: SequenceLengths,
            current_token_index: int,
            cache: Optional[List[torch.Tensor]],
            cache_index: Optional[int],
            output_sequence,
            max_decoder_seq_len,
            temperature,
            incremental: bool,
            use_cache: bool = True,
            encoder_cache: Optional[torch.Tensor] = None,
    ):
        next_token, updated_cache, encoder_cache = predict_next_token(
            model,
            next_sequence,
            full_track_sequence,
            sequence_lengths,
            current_token_index,
            incremental,
            temperature,
            use_cache=use_cache,
            cache=cache,
            cache_index=cache_index,
            encoder_cache=encoder_cache
        )
        output_sequence = next_token if output_sequence is None else torch.cat((output_sequence, next_token), dim=2)
        next_sequence = update_next_sequence(
            next_sequence,
            next_token,
            max_decoder_seq_len,
            current_token_index,
        ) if use_cache is False else next_token.to(next_sequence.device)
        return next_sequence, output_sequence, updated_cache, encoder_cache

    def get_initial_next_sequence(sequence: Optional[torch.Tensor], max_seq_len: int, padding_value: int, codebooks: int, device: Device, pad_left=False, prepend_sos_token=False):
        start_of_sequence_token = get_start_of_sequence_token(codebooks).to(device)
        if sequence is None:
            sequence = start_of_sequence_token
        else:
            sequence = sequence.to(device)
            if prepend_sos_token is True:
                sequence = torch.cat([start_of_sequence_token, sequence], dim=-1)
        
        interleaved_sequence = apply_interleaving(sequence, padding_value)
        sequence_length = sequence.shape[-1]
        padded_sequence = pad_sequence(interleaved_sequence, max_seq_len, padding_value, pad_left)

        if sequence_length > max_seq_len:
            return cut_sequence(padded_sequence, max_seq_len, cut_left=True), max_seq_len

        return padded_sequence, sequence_length
    
    model.eval()
    sliding_window_size = get_params()['sliding_window_size']
    codebooks = get_params()['codebooks']
    max_seq_len = get_params()['max_seq_len']
    max_decoder_seq_len = get_params()['max_decoder_seq_len']
    
    output_sequence = None
    # TODO: Allow prediction to start from a pre-existing full track sequence
    # initial_token_index = initial_sequence.shape[-1] - 1
    initial_token_index = 0

    torch.set_printoptions(edgeitems=20)
    print('--- Input ---')
    print(input_sequence.shape)
    print(input_sequence)
    full_track_sequence, full_track_sequence_length = get_initial_next_sequence(input_sequence, max_seq_len, padding_value, codebooks, device, pad_left=True)
    stem_sequence_length = max_decoder_seq_len
    print('--- Conditioning ---')
    print(full_track_sequence.shape)
    print(full_track_sequence)
    next_sequence, _ = get_initial_next_sequence(None, max_decoder_seq_len, padding_value, codebooks, device)
    sequence_lengths = SequenceLengths(full_track=[full_track_sequence_length], stem=[stem_sequence_length])
    print('--- Next ---')
    print(next_sequence.shape)
    print(next_sequence)
    max_output_tokens = math.ceil(max_output_length_in_seconds * frame_rate)

    loop = tqdm(range(max_output_tokens)).iterable if show_progress else range(max_output_tokens)
    cache: Optional[List[torch.Tensor]] = None
    encoder_cache: Optional[torch.Tensor] = None

    for iteration in loop:
        current_token_index = min(initial_token_index + iteration, max_seq_len - 1)
        cache_index = min(initial_token_index + iteration, sliding_window_size - 1)
        next_sequence, new_output_sequence, cache, encoder_cache = process_next_note_sequence(
            model,
            next_sequence,
            full_track_sequence,
            sequence_lengths,
            current_token_index,
            cache,
            cache_index if use_cache else None,
            output_sequence,
            max_decoder_seq_len,
            temperature,
            incremental=iteration != 0,
            use_cache=use_cache,
            encoder_cache=encoder_cache,
        )
        output_sequence = new_output_sequence

        print('output')
        print(output_sequence.shape)
        print(output_sequence)

    return remove_special_tokens(revert_interleaving(cast(torch.Tensor, output_sequence)), get_special_tokens())


def complete_audio_file(
    model: torch.nn.Module,
    device: Device,
    audio_file: str,
    padding_value: int = get_pad_token_id(),
    max_output_length_in_seconds: float = DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE
) -> AudioSegment:
    wav, sr = torchaudio.load(audio_file)
    output_wav, output_sr = complete_audio(
        model,
        device,
        wav,
        sr,
        padding_value,
        max_output_length_in_seconds,
        temperature,
    )
    temp_file = generate_random_filename(extension='.wav')
    save_audio(output_wav, temp_file, sample_rate=output_sr)

    segment = AudioSegment.from_file(temp_file)
    os.remove(temp_file)

    return segment


def complete_audio(
    model: torch.nn.Module,
    device: Device,
    audio_waveform: torch.Tensor,
    audio_sr: int,
    padding_value: int = get_pad_token_id(),
    max_output_length_in_seconds: float = DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE
) -> Tuple[torch.Tensor, int]:
    input_sequence, frame_rate = encode(audio_waveform, audio_sr, device, add_start_and_end_tokens=False)
    output_sequence = complete_sequence(
        model,
        device,
        input_sequence[0],
        frame_rate=frame_rate,
        max_output_length_in_seconds=max_output_length_in_seconds,
        temperature=temperature,
        padding_value=padding_value
    )
    output_audio, sr = decode(output_sequence, device)
    return output_audio, sr
    