import pytest
import torch
from typing import Tuple
from gigmate.domain.prediction import apply_interleaving, get_initial_next_sequence, update_next_sequence
from gigmate.utils.constants import get_pad_token_id
from gigmate.utils.sequence_utils import get_start_of_sequence_token, pad_sequence


CODEBOOKS = 4
PADDING_VALUE = get_pad_token_id()
device = 'cpu'


def get_sample_data(B: int, K: int, T: int) -> torch.Tensor:
    sequence = torch.zeros((B, K, T), dtype=torch.int)

    for k in range(K):
        sequence[:, k] = torch.arange(1, T + 1, 1, dtype=torch.int)

    return sequence


@pytest.fixture
def test_sequence() -> Tuple[torch.Tensor, int, int, int]:
    B, K, T = 2, CODEBOOKS, 9
    sequence = get_sample_data(B, K, T)
    return sequence, B, K, T


def test_update_next_sequence(test_sequence):
    sequence, B, K, T = test_sequence
    interleaved_sequence = apply_interleaving(sequence, 0)
    current_token = torch.tensor([[[99], [99], [99], [99]]])
    token_index = 5

    next_sequence = update_next_sequence(interleaved_sequence, current_token, 20, token_index)

    assert next_sequence.shape == interleaved_sequence.shape
    assert torch.all(next_sequence[:, :, token_index: token_index + 1] == torch.tensor([
        [99],
        [99],
        [99],
        [99],
    ]).unsqueeze(0).repeat(B, 1, 1)), f"Elements at position {token_index} of the sequence have not been updated"

    assert torch.all(
        torch.cat([next_sequence[:, :, :token_index], next_sequence[:, :, token_index + 1:]], dim=-1) == torch.cat([interleaved_sequence[:, :, :token_index], interleaved_sequence[:, :, token_index + 1:]], dim=-1)
    ), "Elements at other positions should remain the same"


def test_update_next_sequence_when_position_equals_max_seq_len(test_sequence):
    sequence, B, K, T = test_sequence
    interleaved_sequence = apply_interleaving(sequence, 0)
    current_token = torch.tensor([[[99], [99], [99], [99]]])
    sequence_length = interleaved_sequence.shape[-1]
    index = sequence_length
    last_element = interleaved_sequence[0, :, -1]

    next_sequence = update_next_sequence(interleaved_sequence, current_token, sequence_length, index)

    assert next_sequence.shape == (B, K, sequence_length), "Output shape is incorrect"

    assert torch.all(next_sequence[:, :, -2:] == torch.tensor([
        [last_element[0], 99],
        [last_element[1], 99],
        [last_element[2], 99],
        [last_element[3], 99],
    ]).unsqueeze(0).repeat(B, 1, 1)), "Last elements of the sequence are incorrect"


def test_update_next_sequence_when_position_equals_last_element(test_sequence):
    sequence, B, K, T = test_sequence
    interleaved_sequence = apply_interleaving(sequence, 0)
    current_token = torch.tensor([[[99], [99], [99], [99]]])
    sequence_length = interleaved_sequence.shape[2]
    index = sequence_length - 1
    second_last_element = interleaved_sequence[0, :, -2]

    next_sequence = update_next_sequence(interleaved_sequence, current_token, sequence_length, index)

    assert next_sequence.shape == (B, K, sequence_length), "Output shape is incorrect"

    assert torch.all(next_sequence[:, :, -2:] == torch.tensor([
        [second_last_element[0], 99],
        [second_last_element[1], 99],
        [second_last_element[2], 99],
        [second_last_element[3], 99],
    ]).unsqueeze(0).repeat(B, 1, 1)), "Last elements of the sequence are incorrect"


def test_get_initial_next_sequence_with_empty_sequence():
    initial_sequence, seq_len = get_initial_next_sequence(torch.empty((1, CODEBOOKS, 0)), CODEBOOKS, PADDING_VALUE, CODEBOOKS, device, prepend_sos_token=True)
    expected_sequence = apply_interleaving(get_start_of_sequence_token(CODEBOOKS), PADDING_VALUE)

    assert torch.equal(initial_sequence, expected_sequence), "Should return interleaved sequence containing start token only"
    assert seq_len == CODEBOOKS, "Returned sequence length should be equal to number of codebooks (because of interleaving)"


def test_get_initial_next_sequence(test_sequence: torch.Tensor) -> None:
    max_seq_len = 12
    sequence, B, K, T = test_sequence
    initial_sequence, seq_len = get_initial_next_sequence(sequence, max_seq_len, PADDING_VALUE, CODEBOOKS, device)
    expected_sequence = pad_sequence(apply_interleaving(sequence, PADDING_VALUE), max_seq_len, PADDING_VALUE, pad_left=False)

    assert initial_sequence.shape == expected_sequence.shape
    assert torch.equal(initial_sequence, expected_sequence), "Sequence different from expected"
    assert seq_len == T + CODEBOOKS - 1, "Returned sequence length is not correct"


def test_get_initial_next_sequence_with_pad_left(test_sequence: torch.Tensor) -> None:
    max_seq_len = 12
    sequence, B, K, T = test_sequence
    initial_sequence, seq_len = get_initial_next_sequence(sequence, max_seq_len, PADDING_VALUE, CODEBOOKS, device, pad_left=True)
    expected_sequence = pad_sequence(apply_interleaving(sequence, PADDING_VALUE), max_seq_len, PADDING_VALUE, pad_left=True)

    assert initial_sequence.shape == expected_sequence.shape
    assert torch.equal(initial_sequence, expected_sequence), "Sequence different from expected"
    assert seq_len == T + CODEBOOKS - 1, "Returned sequence length is not correct"


if __name__ == "__main__":
    pytest.main([__file__])