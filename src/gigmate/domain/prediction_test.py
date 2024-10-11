import pytest
import torch
from typing import Tuple
from gigmate.domain.prediction import apply_interleaving, cut_sequence_to_the_left, pad_sequence, revert_interleaving, update_interleaved_sequence, update_next_sequence


def get_sample_data(B: int, K: int, T: int) -> torch.Tensor:
    sequence = torch.zeros((B, K, T), dtype=torch.int)

    for k in range(K):
        sequence[:, k] = torch.arange(1, T+1, 1, dtype=torch.int)

    return sequence


@pytest.fixture
def test_sequence() -> Tuple[torch.Tensor, int, int, int]:
    B, K, T = 2, 4, 9
    sequence = get_sample_data(B, K, T)
    return sequence, B, K, T


def test_apply_interleaving(test_sequence):
    sequence, B, K, T = test_sequence
    
    interleaved = apply_interleaving(sequence, 0)
    
    assert interleaved.shape == (B, K, T + K - 1), "Output shape is incorrect"

    assert torch.all(interleaved[:, :, :K] == torch.tensor([
        [1, 2, 3, 4],
        [0, 1, 2, 3],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ]).unsqueeze(0).repeat(B, 1, 1)), "First K elements of the sequence are incorrect"

    assert torch.all(interleaved[:, :, -K:] == torch.tensor([
        [9, 0, 0, 0],
        [8, 9, 0, 0],
        [7, 8, 9, 0],
        [6, 7, 8, 9]
    ]).unsqueeze(0).repeat(B, 1, 1)), "Last K elements of the sequence are incorrect"


def test_revert_interleaving(test_sequence):
    sequence, B, K, T = test_sequence
    
    interleaved = apply_interleaving(sequence, 0)
    reverted = revert_interleaving(interleaved)
    
    assert reverted.shape == sequence.shape, "Reverted shape doesn't match initial shape"
    assert torch.all(reverted == sequence), "Reverted sequence doesn't match initial sequence"


def test_custom_padding_value():
    B, K, T = 2, 3, 5
    sequence = get_sample_data(B, K, T)
    padding_value = -1
    
    interleaved = apply_interleaving(sequence, padding_value)
    assert torch.all(interleaved[:, :, :K] == torch.tensor([
        [ 1,  2,  3],
        [-1,  1,  2],
        [-1, -1,  1],
    ]).unsqueeze(0).repeat(B, 1, 1)), "Custom padding value not applied correctly"


def test_edge_cases():
    # Test with K=1 (no interleaving)
    sequence = torch.randn(2, 1, 10)
    interleaved = apply_interleaving(sequence, 0)
    assert torch.all(interleaved == sequence), "K=1 case failed"
    
    # Test with T=1 (minimal sequence length)
    sequence = torch.randn(2, 3, 1)
    interleaved = apply_interleaving(sequence, 0)
    reverted = revert_interleaving(interleaved)
    assert torch.all(reverted == sequence), "T=1 case failed"


def test_large_input():
    B, K, T = 10, 20, 1000
    sequence = torch.randn(B, K, T)
    interleaved = apply_interleaving(sequence, 0)
    reverted = revert_interleaving(interleaved)
    assert torch.all(torch.isclose(reverted, sequence, atol=1e-6)), "Large input test failed"


def test_dtype_and_device_preservation():
    sequence = torch.randn(2, 3, 5, dtype=torch.float64)
    if torch.cuda.is_available():
        sequence = sequence.cuda()
    
    interleaved = apply_interleaving(sequence, 0)
    assert interleaved.dtype == sequence.dtype, "Data type not preserved"
    assert interleaved.device == sequence.device, "Device not preserved"
    
    reverted = revert_interleaving(interleaved)
    assert reverted.dtype == sequence.dtype, "Data type not preserved after reversion"
    assert reverted.device == sequence.device, "Device not preserved after reversion"


def test_update_interleaved_sequence(test_sequence):
    sequence, B, K, T = test_sequence
    sequence = apply_interleaving(sequence, 0)
    position = 4

    interleaved = update_interleaved_sequence(sequence, position, torch.tensor([[[-5], [-5], [-5], [-5]]]), padding_value=0)

    assert interleaved.shape == (B, K, T + K -1), "Output shape is incorrect"

    assert torch.all(interleaved[:, :, 4:8] == torch.tensor([
        [-5, 6, 7, 8],
        [ 4,-5, 6, 7],
        [ 3, 4,-5, 6],
        [ 2, 3, 4,-5]
    ]).unsqueeze(0).repeat(B, 1, 1)), "Last K elements of the sequence are incorrect"


def test_update_interleaved_sequence_when_position_equals_last_element(test_sequence):
    sequence, B, K, T = test_sequence
    sequence = apply_interleaving(sequence, 0)
    position = 9

    interleaved = update_interleaved_sequence(sequence, position, torch.tensor([[[10], [10], [10], [10]]]), padding_value=0)
    
    assert interleaved.shape == (B, K, position + K), "Output shape is incorrect"

    assert torch.all(interleaved[:, :, -K:] == torch.tensor([
        [10, 0, 0, 0],
        [ 9,10, 0, 0],
        [ 8, 9,10, 0],
        [ 7, 8, 9,10]
    ]).unsqueeze(0).repeat(B, 1, 1)), "Last K elements of the sequence are incorrect"


def test_cut_sequence_to_length(test_sequence):
    sequence, B, K, T = test_sequence
    cut_sequence = cut_sequence_to_the_left(sequence, 3)
    
    expected_tensor = torch.tensor([
        [7, 8, 9],
        [7, 8, 9],
        [7, 8, 9],
        [7, 8, 9]
    ]).unsqueeze(0).repeat(B, 1, 1)

    assert cut_sequence.shape == (B, K, 3), "Output shape is incorrect"
    assert torch.all(cut_sequence == expected_tensor), "Sequence elements are incorrect"


def test_cut_sequence_to_length_when_length_equals_sequence_length(test_sequence):
    sequence, B, K, T = test_sequence
    cut_sequence = cut_sequence_to_the_left(sequence, T)
    
    assert cut_sequence.shape == (B, K, T), "Output shape is incorrect"
    assert torch.all(cut_sequence == sequence), "Sequence elements are incorrect"


def test_cut_sequence_to_length_when_length_greater_than_sequence_length(test_sequence):
    sequence, B, K, T = test_sequence
    cut_sequence = cut_sequence_to_the_left(sequence, T + 1)
    
    assert cut_sequence.shape == (B, K, T), "Output shape is incorrect"
    assert torch.all(cut_sequence == sequence), "Sequence elements are incorrect"


def test_update_next_sequence(test_sequence):
    sequence, B, K, T = test_sequence
    interleaved_sequence = apply_interleaving(sequence, 0)
    current_token = torch.tensor([[[99], [99], [99], [99]]])

    next_sequence = update_next_sequence(interleaved_sequence, current_token, 20, 4, 0)

    assert next_sequence.shape == interleaved_sequence.shape
    assert torch.all(next_sequence[:, :, 4:8] == torch.tensor([
        [99, 6, 7, 8],
        [ 4,99, 6, 7],
        [ 3, 4,99, 6],
        [ 2, 3, 4,99]
    ]).unsqueeze(0).repeat(B, 1, 1)), "Last K elements of the sequence are incorrect"


def test_update_next_sequence_when_position_equals_last_element(test_sequence):
    sequence, B, K, T = test_sequence
    interleaved_sequence = apply_interleaving(sequence, 0)
    current_token = torch.tensor([[[99], [99], [99], [99]]])
    sequence_length = interleaved_sequence.shape[2]
    last_element = sequence_length - 1

    next_sequence = update_next_sequence(interleaved_sequence, current_token, sequence_length, last_element, 0)

    assert next_sequence.shape == (B, K, sequence_length), "Output shape is incorrect"

    assert torch.all(next_sequence[:, :, -K:] == torch.tensor([
        [99, 0, 0, 0],
        [ 0,99, 0, 0],
        [ 0, 0,99, 0],
        [ 9, 0, 0,99]
    ]).unsqueeze(0).repeat(B, 1, 1)), "Last K elements of the sequence are incorrect"


def test_update_next_sequence_when_position_equals_second_last_element(test_sequence):
    sequence, B, K, T = test_sequence
    interleaved_sequence = apply_interleaving(sequence, 0)
    current_token = torch.tensor([[[99], [99], [99], [99]]])
    sequence_length = interleaved_sequence.shape[2]
    last_element = sequence_length - 2

    next_sequence = update_next_sequence(interleaved_sequence, current_token, sequence_length, last_element, 0)

    assert next_sequence.shape == (B, K, sequence_length), "Output shape is incorrect"

    assert torch.all(next_sequence[:, :, -K:] == torch.tensor([
        [99, 0, 0, 0],
        [ 0,99, 0, 0],
        [ 9, 0,99, 0],
        [ 8, 9, 0,99]
    ]).unsqueeze(0).repeat(B, 1, 1)), "Last K elements of the sequence are incorrect"


def test_pad_sequence(test_sequence):
    sequence, B, K, T = test_sequence
    padded_sequence = pad_sequence(sequence, 32, 0)
    
    assert padded_sequence.shape == (B, K, 32), "Invalid shape after padding"
    assert padded_sequence[:, :, T:].sum() == 0, "Padding elements are not zeros"


def test_pad_sequence_when_length_equals_sequence_length(test_sequence):
    sequence, B, K, T = test_sequence
    padded_sequence = pad_sequence(sequence, T, 0)
    
    assert padded_sequence.shape == (B, K, T), "Invalid shape after padding"
    assert torch.equal(padded_sequence, sequence), "Sequence has been modified"


if __name__ == "__main__":
    pytest.main([__file__])