import math
from typing import List, Optional, Tuple, cast
from gigmate.model.codec import decode, encode
from encodec.utils import save_audio
import torch
from tqdm import tqdm
from gigmate.utils.audio_utils import generate_random_filename
from gigmate.utils.constants import get_pad_token_id, get_params, get_start_of_sequence_token_id
import torch.nn.functional as F
import torch._dynamo as dynamo

DEFAULT_MAX_OUTPUT_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS = 20

def sample_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # If temp is 0 then next_token is the argmax of logits
    if temperature == 0.0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    # If temp is not 0 then next_token is sampled out of logits
    else:
        logits = logits / temperature
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

    return next_token

def predict_next_token(model: torch.nn.Module, input: torch.Tensor, current_token_index: int, incremental: bool, temperature: float = DEFAULT_TEMPERATURE, use_cache: bool = True, cache: Optional[List[torch.Tensor]] = None, cache_index: Optional[int] = None) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    with torch.inference_mode():

        #explanation = dynamo.explain(model)(input, use_cache=use_cache, current_token_index=current_token_index if incremental else None)
        #print(explanation)

        outputs, updated_cache = model(input, use_cache=use_cache, cache=cache, cache_index=cache_index if incremental else None)
        outputs = outputs.squeeze(0).transpose(0, 1) # switch codebooks and sequence dimensions
        outputs = outputs[-1 if use_cache and incremental else current_token_index] # remove batch dimension and take only next token logits
    
    predicted_tokens = sample_from_logits(outputs, temperature).squeeze(-1) # sample and remove last dimensio
    return predicted_tokens.detach().to('cpu'), updated_cache

def get_all_program_change_tokens(tokens_to_ids: dict[str, list[int]], except_for_program: Optional[int] = None) -> list[int]:
    return [id for token, ids in tokens_to_ids.items() for id in ids if token.startswith('Program') and not token == f'Program_{except_for_program}']

def apply_interleaving(sequence: torch.Tensor, padding_value: int) -> torch.Tensor:
    """
    Applies delayed interleaving to the input sequence.

    This function takes an input sequence and applies a delayed interleaving pattern,
    padding with a specified value where necessary.

    CB0 1 2 3 4 5 6 7 8 9 0 0 0
    CB1 0 1 2 3 4 5 6 7 8 9 0 0
    CB2 0 0 1 2 3 4 5 6 7 8 9 0
    CB3 0 0 0 1 2 3 4 5 6 7 8 9

    Args:
        sequence (torch.Tensor): The input sequence to be interleaved.
            Shape: (B, K, T), where:
            - B is the batch size
            - K is the number of codebooks
            - T is the original sequence length
        padding_value (int): The value to use for padding.

    Returns:
        torch.Tensor: The interleaved sequence.
            Shape: (B, K, T + K - 1)

    Example:
        >>> B, K, T = 2, 4, 9
        >>> sequence = torch.arange(1, B*K*T + 1).reshape(B, K, T)
        >>> interleaved = apply_interleaving(initial_sequence, padding_value=-1)
        >>> print(interleaved.shape)
        torch.Size([2, 4, 12])
    """
    B, K, T = sequence.shape
    
    # Create the output tensor with shape (B, K, T + K - 1)
    output = torch.full((B, K, T + K - 1), fill_value=padding_value, dtype=sequence.dtype, device=sequence.device)
    
    for k in range(K):
        # Shift each codebook by its index
        output[:, k, k:k+T] = sequence[:, k, :]
    
    return output

def revert_interleaving(sequence: torch.Tensor) -> torch.Tensor:
    """
    Reverts a delayed interleaved sequence back to its original form.

    This function takes a sequence that has been processed with delayed interleaving
    and reverts it to its original form. It is the inverse operation of 
    `apply_interleaving`.

    Args:
        sequence (torch.Tensor): The interleaved input sequence.
            Shape: (B, K, T + K - 1), where:
            - B is the batch size
            - K is the number of codebooks
            - T + K - 1 is the extended sequence length

    Returns:
        torch.Tensor: The reverted sequence.
            Shape: (B, K, T), where T is the original sequence length.

    Example:
        >>> B, K, T = 2, 4, 9
        >>> interleaved = torch.randn(B, K, T + K - 1)
        >>> original = revert_interleaving(interleaved)
        >>> print(original.shape)
        torch.Size([2, 4, 9])
    """
    B, K, T_extended = sequence.shape
    original_t = T_extended - K + 1
    
    # Create the output tensor with shape (B, K, T)
    output = torch.zeros((B, K, original_t), dtype=sequence.dtype, device=sequence.device)

    for k in range(K):
        # Extract the original sequence for each codebook
        output[:, k, :] = sequence[:, k, k:k+original_t]

    return output

def update_interleaved_sequence(sequence: torch.Tensor, position: int, new_tokens: torch.Tensor, padding_value: int) -> torch.Tensor:
    """
    Updates the tokens at a specific position in the input sequence, taking into consideration delayed interleaving.
    If the original sequence is:
    
    CB0 1 2 3 4 0 0
    CB0 0 1 2 3 4 0
    CB0 0 0 1 2 3 4

    and the arguments are 4 and [5, 5, 5], the output will be:

    CB0 1 2 3 4 5 0 0
    CB0 0 1 2 3 4 5 0
    CB0 0 0 1 2 3 4 5
    """

    original_sequence_length = sequence.shape[2]
    new_sequence_length = max(original_sequence_length, position + new_tokens.shape[1])
    sequence = torch.cat([sequence, torch.full((sequence.shape[0], sequence.shape[1], new_sequence_length - original_sequence_length), padding_value, dtype=sequence.dtype, device=sequence.device)], dim=2)

    for k in range(new_tokens.shape[1]):
        sequence[:,k,position+k] = new_tokens.squeeze(0).squeeze(-1)[k]
    
    return sequence

def cut_sequence_to_the_left(sequence: torch.Tensor, length: int) -> torch.Tensor:
    """
    Cut the specified sequence keeping only the *length* element on the right.

    Returns:
        torch.Tensor: the cut sequence
    """
    return sequence[:, :, -length:]

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
        return cut_sequence_to_the_left(interleaved_sequence, max_seq_len)
    
    return interleaved_sequence

def pad_sequence(sequence: torch.Tensor, max_len: int, padding_value: int):
    B, K, T = sequence.shape

    if T >= max_len:
        return sequence

    pad_length = max_len - T
    padding = torch.tensor([padding_value]).repeat(B * K * pad_length).reshape((B, K, pad_length)).to(sequence.device)
    return torch.cat([sequence, padding], dim=-1)

def complete_sequence(
    model: torch.nn.Module,
    device: str,
    input_sequence: torch.Tensor,
    frame_rate: int,
    padding_value: int,
    verbose: bool = False,
    max_output_length_in_seconds: float = DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
    show_progress: bool = False,
    use_cache: bool = True,
) -> torch.Tensor:

    def process_next_note_sequence(model, next_sequence, current_token_index: int, cache: Optional[List[torch.Tensor]], cache_index: int, output_sequence, max_seq_len, temperature, padding_value: int, incremental: bool, use_cache: bool = True):
        next_token, updated_cache = predict_next_token(model, next_sequence, current_token_index, incremental, temperature, use_cache=use_cache, cache=cache, cache_index=cache_index)
        reshaped_next_token = next_token.unsqueeze(0).unsqueeze(2)
        output_sequence = reshaped_next_token if output_sequence is None else torch.cat((output_sequence, reshaped_next_token), dim=2)
        next_sequence = update_next_sequence(next_sequence, reshaped_next_token, max_seq_len, current_token_index, padding_value=padding_value) if use_cache == False or incremental is None else reshaped_next_token.to(next_sequence.device)
        return next_sequence, output_sequence, updated_cache

    def get_initial_next_sequence(sequence: torch.Tensor, max_seq_len: int, padding_value: int):
        _, K, sequence_length = sequence.shape
        start_of_sequence_token = torch.full((1, K, 1), get_start_of_sequence_token_id()).to(sequence.device)
        sequence = torch.cat([start_of_sequence_token, sequence], dim=-1)
        interleaved_sequence = apply_interleaving(sequence, padding_value)
        padded_sequence = pad_sequence(interleaved_sequence, max_seq_len, padding_value)

        if sequence_length > max_seq_len:
            return cut_sequence_to_the_left(padded_sequence, max_seq_len)

        return padded_sequence
    

    model.eval()
    initial_sequence = input_sequence.clone().detach()
    sliding_window_size = get_params()['sliding_window_size']
    max_seq_len = get_params()['max_seq_len']
    
    output_sequence = None
    initial_token_index = initial_sequence.shape[-1] - 1

    next_sequence = get_initial_next_sequence(initial_sequence, max_seq_len, padding_value).to(device)
    max_output_tokens = math.ceil(max_output_length_in_seconds * frame_rate)

    loop = tqdm(range(max_output_tokens)) if show_progress else range(max_output_tokens)
    cache: Optional[List[torch.Tensor]] = None

    for iteration in loop:
        current_token_index = min(initial_token_index + iteration, max_seq_len - 1)
        cache_index = min(initial_token_index + iteration, sliding_window_size - 1)
        next_sequence, new_output_sequence, cache = process_next_note_sequence(model, next_sequence, current_token_index, cache, cache_index, output_sequence, max_seq_len, temperature, padding_value=padding_value, incremental=iteration != 0, use_cache=use_cache)
        output_sequence = new_output_sequence

    return revert_interleaving(cast(torch.Tensor, output_sequence))

def complete_audio(
    model: torch.nn.Module,
    device: str,
    audio_file: str,
    max_seq_len: int,
    verbose: bool = False,
    padding_value: int = get_pad_token_id(),
    max_output_length_in_seconds: float = DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:
    input_sequence, frame_rate = encode(audio_file)
    output_sequence = complete_sequence(model, device, input_sequence, frame_rate=frame_rate, verbose=verbose, max_output_length_in_seconds=max_output_length_in_seconds, temperature=temperature, padding_value=padding_value)
    output_audio = decode(output_sequence)
    temp_file = generate_random_filename(extension='.wav')
    save_audio(output_audio, temp_file, sample_rate=24000)

    return temp_file
