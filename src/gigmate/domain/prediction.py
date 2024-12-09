import math
from typing import List, Optional, Tuple, cast
from pydub import AudioSegment
import torchaudio
from encodec.utils import save_audio
import torch
from tqdm import tqdm
import os

from gigmate.dataset.dataset import SequenceLengths
from gigmate.domain.sampling import sample_from_logits
from gigmate.model.codec import decode, encode
from gigmate.utils.audio_utils import generate_random_filename
from gigmate.utils.constants import CODEBOOKS, get_pad_token_id, get_params, get_start_of_sequence_token_id
from gigmate.utils.device import Device
from gigmate.utils.sequence_utils import apply_interleaving, cut_sequence, get_start_of_sequence_token, pad_sequence, revert_interleaving, shift_sequence

DEFAULT_TEMPERATURE = 0
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
        predicted_tokens = sample_from_logits(outputs, temperature, no_special_tokens=False).unsqueeze(0).unsqueeze(2)  # sample and remove last dimension
        
    return predicted_tokens.detach().to('cpu'), updated_cache, updated_encoder_cache


# 0 1 2 3
# S 1 2 3
# P S 1 2
# P P S 1
# P P P S
def correct_current_token(token: torch.Tensor, index: int) -> torch.Tensor:
    if index == 1:
        # Correct second token
        token[:, 1, :] = get_start_of_sequence_token_id()
        token[:, 2, :] = get_pad_token_id()
        token[:, 3, :] = get_pad_token_id()

    if index == 2:
        # Correct second token
        token[:, 2, :] = get_start_of_sequence_token_id()
        token[:, 3, :] = get_pad_token_id()

    if index == 3:
        # Correct second token
        token[:, 3, :] = get_start_of_sequence_token_id()

    return token


def update_next_sequence(previous_next_sequence: torch.Tensor, current_token: torch.Tensor, max_seq_len: int, index: int):
    """
    Updates the next sequence with the current token, taking into account interleaving and maximum sequence length.

    Returns:
        torch.Tensor: the updated next sequence ready for the next inference iteration
    """
    current_token = current_token.to(previous_next_sequence.device)
    current_token = correct_current_token(current_token, index)

    assert index <= max_seq_len, "index must not be larger than max_seq_len"

    if index == max_seq_len:
        next_sequence = shift_sequence(previous_next_sequence)
        index = max_seq_len - 1  # Update index after shifting
    else:
        next_sequence = previous_next_sequence

    next_sequence[:, :, index] = current_token.reshape((1, 4))
    
    return next_sequence


def get_initial_next_sequence(sequence: torch.Tensor, max_seq_len: int, padding_value: int, codebooks: int, device: Device, pad_left=False, prepend_sos_token=False, interleaving: bool = True):
    start_of_sequence_token = get_start_of_sequence_token(codebooks).to(device)

    sequence = sequence.to(device)
    if prepend_sos_token is True:
        sequence = torch.cat([start_of_sequence_token, sequence], dim=-1)

    if interleaving is True:
        sequence = apply_interleaving(sequence, padding_value)
    sequence_length = sequence.shape[-1]
    padded_sequence = pad_sequence(sequence, max_seq_len, padding_value, pad_left)

    if sequence_length > max_seq_len:
        return cut_sequence(padded_sequence, max_seq_len, cut_left=True), max_seq_len

    return padded_sequence, sequence_length


def complete_sequence(
    model: torch.nn.Module,
    device: Device,
    full_track_sequence: torch.Tensor,
    frame_rate: float,
    padding_value: int,
    max_output_length_in_seconds: float = DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
    show_progress: bool = False,
    use_cache: bool = True,
    input_sequence: torch.Tensor = torch.empty((1, CODEBOOKS, 0), dtype=torch.int64)
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
            current_token_index + 1,
        ) if use_cache is False else next_token.to(next_sequence.device)
        return next_sequence, output_sequence, updated_cache, encoder_cache
    
    model.eval()
    sliding_window_size = get_params()['sliding_window_size']
    codebooks = get_params()['codebooks']
    max_seq_len = get_params()['max_seq_len']
    max_decoder_seq_len = get_params()['max_decoder_seq_len']
    
    output_sequence = None
    full_track_initial_sequence, full_track_initial_sequence_length = get_initial_next_sequence(full_track_sequence, max_seq_len, padding_value, codebooks, device, pad_left=True, interleaving=False)
    stem_sequence_length = max_decoder_seq_len
    next_sequence, _ = get_initial_next_sequence(input_sequence, max_decoder_seq_len, padding_value, codebooks, device, prepend_sos_token=True)
    initial_token_index = input_sequence.shape[-1] - 1 if input_sequence.shape[-1] > 0 else 0
    sequence_lengths = SequenceLengths(full_track=[full_track_initial_sequence_length], stem=[stem_sequence_length])
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
            full_track_initial_sequence,
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

    return revert_interleaving(cast(torch.Tensor, output_sequence))


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
    