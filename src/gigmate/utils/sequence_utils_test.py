import pytest
import torch
from typing import Tuple

from gigmate.utils.constants import RANDOM_SEED, get_pad_token_id, get_special_tokens
from gigmate.utils.sequence_utils import apply_interleaving, cut_sequence, get_end_of_sequence_token, get_start_of_sequence_token, pad_sequence
from gigmate.utils.sequence_utils import remove_special_tokens_from_target_and_logits, revert_interleaving, add_start_and_end_tokens, mix_sequences


def get_sample_data(B: int, K: int, T: int) -> torch.Tensor:
    sequence = torch.zeros((B, K, T), dtype=torch.int)

    for k in range(K):
        sequence[:, k] = torch.arange(1, T + 1, 1, dtype=torch.int)

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
        [1, 2, 3],
        [-1, 1, 2],
        [-1, -1, 1],
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


def test_cut_sequence_to_length(test_sequence):
    sequence, B, K, T = test_sequence
    cut_seq = cut_sequence(sequence, 3, cut_left=True)
    
    expected_tensor = torch.tensor([
        [7, 8, 9],
        [7, 8, 9],
        [7, 8, 9],
        [7, 8, 9]
    ]).unsqueeze(0).repeat(B, 1, 1)

    assert cut_seq.shape == (B, K, 3), "Output shape is incorrect"
    assert torch.all(cut_seq == expected_tensor), "Sequence elements are incorrect"


def test_cut_sequence_to_length_when_length_equals_sequence_length(test_sequence):
    sequence, B, K, T = test_sequence
    cut_seq = cut_sequence(sequence, T, cut_left=True)
    
    assert cut_seq.shape == (B, K, T), "Output shape is incorrect"
    assert torch.all(cut_seq == sequence), "Sequence elements are incorrect"


def test_cut_sequence_to_length_when_length_greater_than_sequence_length(test_sequence):
    sequence, B, K, T = test_sequence
    cut_seq = cut_sequence(sequence, T + 1, cut_left=True)
    
    assert cut_seq.shape == (B, K, T), "Output shape is incorrect"
    assert torch.all(cut_seq == sequence), "Sequence elements are incorrect"
    

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


def test_remove_special_tokens():
    targets = torch.tensor([
        [
            [1, 2, 3, 4, get_pad_token_id(), 0, get_pad_token_id()],
            [1, 2, 3, 4, get_pad_token_id(), 0, get_pad_token_id()],
            [1, 2, 3, 4, get_pad_token_id(), 0, get_pad_token_id()],
            [1, 2, 3, 4, get_pad_token_id(), 0, get_pad_token_id()],
        ],
    ])
    logits = torch.tensor([
        [
            [1, 2, 3, 4, 4, 1, 1],
            [1, 2, 3, 1, 2, 4, 3],
            [1, 2, 3, 4, 1, 0, 3],
            [1, 2, 3, 4, 0, 0, 4],
        ]
    ])
    targets, logits = remove_special_tokens_from_target_and_logits(targets, logits, get_special_tokens())

    assert targets.shape == logits.shape
    assert targets.shape == (1, 4, 5)


def test_add_start_and_end_token(test_sequence):
    sequence, B, K, T = test_sequence
    padded_sequence = add_start_and_end_tokens(sequence)
    
    assert padded_sequence.shape == (B, K, T + 2), "Invalid shape after adding start and end token"
    assert torch.equal(padded_sequence[:, :, 0:1], get_start_of_sequence_token(K, batch_size=B)), "Start token is not correct"
    assert torch.equal(padded_sequence[:, :, -1:], get_end_of_sequence_token(K, batch_size=B)), "End token is not correct"


def test_mix_sequences_prob_1(test_sequence):
    sequence, B, K, T = test_sequence
    
    second_sequence = torch.randint(10, 20, size=sequence.shape)
    mixed_sequence = mix_sequences(sequence, second_sequence, 1.0)

    assert torch.equal(mixed_sequence, sequence)


def test_mix_sequences_prob_0(test_sequence):
    sequence, B, K, T = test_sequence
    
    second_sequence = torch.randint(10, 20, size=sequence.shape)
    mixed_sequence = mix_sequences(sequence, second_sequence, 0.0)

    assert torch.equal(mixed_sequence, second_sequence)


def test_mix_sequences_prob_0_5(test_sequence):
    sequence, B, K, T = test_sequence
    torch.manual_seed(RANDOM_SEED)
    second_sequence = torch.randint(10, 20, size=sequence.shape)
    mixed_sequence = mix_sequences(sequence, second_sequence, 0.5)

    # At a sequence position, the resulting sequence must have codebooks from the same sequence.
    # We must not mix codebooks from different sequences. 

    expected_sequence = torch.tensor([
        [[12, 2, 3, 4, 5, 15, 10, 14, 9],
         [13, 2, 3, 4, 5, 11, 12, 15, 9],
         [17, 2, 3, 4, 5, 11, 19, 13, 9],
         [19, 2, 3, 4, 5, 15, 19, 13, 9]],

        [[1, 16, 3, 10, 5, 12, 7, 8, 17],
         [1, 13, 3, 13, 5, 10, 7, 8, 19],
         [1, 19, 3, 14, 5, 18, 7, 8, 10],
         [1, 10, 3, 13, 5, 11, 7, 8, 19]]
    ])

    assert torch.equal(mixed_sequence, expected_sequence)
