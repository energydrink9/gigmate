from typing import List, Tuple
from torch import Tensor
import torch
from gigmate.utils.constants import get_end_of_sequence_token_id, get_pad_token_id, get_start_of_sequence_token_id


def pad_sequence(sequence: Tensor, max_len: int, padding_value: int, pad_left: bool = False) -> Tensor:
    B, K, T = sequence.shape

    if T >= max_len:
        return sequence

    pad_length = max_len - T
    padding = torch.tensor([padding_value], device=sequence.device).repeat(B * K * pad_length).reshape((B, K, pad_length))

    # Pad the sequence on the left or right based on the pad_left flag
    if pad_left:
        return torch.cat([padding, sequence], dim=-1)
    else:
        return torch.cat([sequence, padding], dim=-1)


def get_padding_token(codebooks: int) -> Tensor:
    return torch.full((1, codebooks, 1), get_pad_token_id())


def get_start_of_sequence_token(codebooks: int, batch_size: int = 1) -> torch.Tensor:
    return torch.full((batch_size, codebooks, 1), get_start_of_sequence_token_id())


def get_end_of_sequence_token(codebooks: int, batch_size: int = 1) -> torch.Tensor:
    return torch.full((batch_size, codebooks, 1), get_end_of_sequence_token_id())


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
        output[:, k, k: k + T] = sequence[:, k, :]
    
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
        output[:, k, :] = sequence[:, k, k: k + original_t]

    return output


def apply_start_tokens_to_interleaved_sequence(sequence: Tensor, codebooks: int, start_token_id: int):
    for codebook in range(codebooks):
        sequence[:, codebook, :codebook + 1] = start_token_id

    return sequence


def update_interleaved_sequence(sequence: Tensor, position: int, new_tokens: Tensor, padding_value: int) -> torch.Tensor:
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
        sequence[:, k, position + k] = new_tokens.squeeze(0).squeeze(-1)[k]
    
    return sequence


def cut_sequence(sequence: torch.Tensor, length: int, cut_left: bool = False) -> torch.Tensor:
    """
    Cut the specified sequence keeping only the *length* element on the left.

    Returns:
        torch.Tensor: the cut sequence
    """

    if cut_left:
        return sequence[:, :, -length:]
    else:
        return sequence[:, :, :length]


def shift_sequence(sequence: torch.Tensor, shifts: int = -1) -> torch.Tensor:
    return torch.roll(sequence, shifts=shifts, dims=-1)


def mask_sequence(sequence: torch.Tensor, length: int, mask_value: int) -> torch.Tensor:
    """
    Mask *length* elements on the right of the specified sequence.

    Returns:
        torch.Tensor: the masked sequence
    """

    sequence[:, :, -length:] = mask_value

    return sequence


def remove_special_tokens_from_target_and_logits(target: Tensor, logit: Tensor, special_tokens: List[int]) -> Tuple[Tensor, Tensor]:
    """
    Remove special tokens from the target sequence and the elements with the corresponding indices from the logits.
    """

    mask = ~torch.isin(target[0, 0, :], torch.tensor(special_tokens, device=target.device))

    filtered_target = target[:, :, mask]
    filtered_logits = logit[:, :, mask]

    return filtered_target, filtered_logits


def remove_special_tokens(sequence: Tensor, special_tokens: List[int]) -> Tensor:
    """
    Remove special tokens from the target sequence and the elements with the corresponding indices from the logits.
    """

    mask = ~torch.isin(sequence[0, 0, :], torch.tensor(special_tokens, device=sequence.device))

    return sequence[:, :, mask]


def add_start_and_end_tokens(sequence: Tensor):
    
    batch_size, codebooks, _ = sequence.shape
    start_token = get_start_of_sequence_token(codebooks, batch_size=batch_size)
    end_token = get_end_of_sequence_token(codebooks, batch_size=batch_size)

    return torch.cat([start_token, sequence, end_token], dim=-1)
