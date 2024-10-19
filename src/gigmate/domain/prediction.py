import math
from typing import List, Optional, Tuple, cast

from pydub import AudioSegment
from gigmate.domain.sampling import sample_from_logits
from gigmate.model.codec import decode, encode_file
from encodec.utils import save_audio
import torch
from tqdm import tqdm
import os

from gigmate.utils.audio_utils import generate_random_filename
from gigmate.utils.constants import get_pad_token_id, get_params
from gigmate.utils.device import Device
from gigmate.utils.sequence_utils import apply_interleaving, cut_sequence, get_start_of_sequence_token, pad_sequence, revert_interleaving, update_interleaved_sequence

DEFAULT_MAX_OUTPUT_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS = 20


def predict_next_token(
        model: torch.nn.Module,
        input: torch.Tensor,
        full_track_sequence: torch.Tensor,
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
            use_cache=use_cache,
            cache=cache,
            cache_index=cache_index if incremental else None,
            encoder_cache=encoder_cache,
        )
        outputs = outputs.squeeze(0).transpose(0, 1)  # switch codebooks and sequence dimensions
        outputs = outputs[-1 if use_cache and incremental else current_token_index]  # remove batch dimension and take only next token logits
    
        predicted_tokens = sample_from_logits(outputs, temperature).unsqueeze(0).unsqueeze(2)  # sample and remove last dimension

    return predicted_tokens.detach().to('cpu'), updated_cache, updated_encoder_cache


def update_next_sequence(previous_next_sequence: torch.Tensor, current_token: torch.Tensor, max_seq_len: int, current_token_index: int, padding_value: int):
    """
    Updates the next sequence with the current token, taking into account interleaving and maximum sequence length.

    Returns:
        torch.Tensor: the updated next sequence ready for the next inference iteration
    """
    current_token = current_token.to(previous_next_sequence.device)
    
    interleaved_sequence = update_interleaved_sequence(previous_next_sequence, current_token_index, current_token, padding_value=padding_value)
    sequence_length = interleaved_sequence.shape[2]

    if sequence_length > max_seq_len:
        return cut_sequence(interleaved_sequence, max_seq_len, cut_left=True)
    
    return interleaved_sequence


def complete_sequence(
    model: torch.nn.Module,
    device: Device,
    input_sequence: torch.Tensor,
    frame_rate: int,
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
            current_token_index: int,
            cache: Optional[List[torch.Tensor]],
            cache_index: Optional[int],
            output_sequence,
            max_decoder_seq_len,
            temperature,
            padding_value: int,
            incremental: bool,
            use_cache: bool = True,
            encoder_cache: Optional[torch.Tensor] = None,
    ):
        next_token, updated_cache, encoder_cache = predict_next_token(model, next_sequence, full_track_sequence, current_token_index, incremental, temperature, use_cache=use_cache, cache=cache, cache_index=cache_index, encoder_cache=encoder_cache)
        output_sequence = next_token if output_sequence is None else torch.cat((output_sequence, next_token), dim=2)
        next_sequence = update_next_sequence(next_sequence, next_token, max_decoder_seq_len, current_token_index, padding_value=padding_value) if use_cache is False or incremental is None else next_token.to(next_sequence.device)
        return next_sequence, output_sequence, updated_cache, encoder_cache

    def get_initial_next_sequence(sequence: Optional[torch.Tensor], max_seq_len: int, padding_value: int, codebooks: int, device: Device, pad_left=False, prepend_sos_token=False):
        start_of_sequence_token = get_start_of_sequence_token(codebooks).to(device)
        if sequence is None:
            sequence = start_of_sequence_token
        else:
            sequence = sequence.to(device)
            if prepend_sos_token is True:
                sequence = torch.cat([start_of_sequence_token], dim=-1)
            
        sequence_length = sequence.shape[-1]
        interleaved_sequence = apply_interleaving(sequence, padding_value)
        padded_sequence = pad_sequence(interleaved_sequence, max_seq_len, padding_value, pad_left)

        if sequence_length > max_seq_len:
            return cut_sequence(padded_sequence, max_seq_len, cut_left=True)

        return padded_sequence
    
    model.eval()
    sliding_window_size = get_params()['sliding_window_size']
    codebooks = get_params()['codebooks']
    max_seq_len = get_params()['max_seq_len']
    max_decoder_seq_len = get_params()['max_decoder_seq_len']
    
    output_sequence = None
    # TODO: Allow prediction to start from a pre-existing sequence
    # initial_token_index = initial_sequence.shape[-1] - 1
    initial_token_index = 0

    full_track_sequence = get_initial_next_sequence(input_sequence, max_seq_len, padding_value, codebooks, device, pad_left=True).to(device)
    next_sequence = get_initial_next_sequence(None, max_decoder_seq_len, padding_value, codebooks, device).to(device)
    max_output_tokens = math.ceil(max_output_length_in_seconds * frame_rate)

    loop = tqdm(range(max_output_tokens)) if show_progress else range(max_output_tokens)
    cache: Optional[List[torch.Tensor]] = None
    encoder_cache: Optional[torch.Tensor] = None

    for iteration in loop:
        current_token_index = min(initial_token_index + iteration, max_seq_len - 1)
        cache_index = min(initial_token_index + iteration, sliding_window_size - 1)
        next_sequence, new_output_sequence, cache, encoder_cache = process_next_note_sequence(
            model,
            next_sequence,
            full_track_sequence,
            current_token_index,
            cache,
            cache_index if use_cache else None,
            output_sequence,
            max_decoder_seq_len,
            temperature,
            padding_value=padding_value,
            incremental=iteration != 0,
            use_cache=use_cache,
            encoder_cache=encoder_cache,
        )
        output_sequence = new_output_sequence

    return revert_interleaving(cast(torch.Tensor, output_sequence))


def complete_audio(
    model: torch.nn.Module,
    device: Device,
    audio_file: str,
    verbose: bool = False,
    padding_value: int = get_pad_token_id(),
    max_output_length_in_seconds: float = DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE
) -> AudioSegment:
    input_sequence, frame_rate = encode_file(audio_file, device)
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
    temp_file = generate_random_filename(extension='.wav')
    save_audio(output_audio, temp_file, sample_rate=sr)

    segment = AudioSegment.from_file(temp_file)
    os.remove(temp_file)

    return segment
