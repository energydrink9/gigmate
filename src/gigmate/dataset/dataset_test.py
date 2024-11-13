

import random
import time
import pytest
import torch
from torch.utils.data import DataLoader

from gigmate.dataset.dataset import get_data_loader, get_inputs_and_targets, get_model_input, restore_initial_sequence
from gigmate.utils.device import Device, get_device
from gigmate.utils.constants import get_pad_token_id, get_random_seed
from gigmate.utils.sequence_utils import apply_interleaving, cut_sequence, get_start_of_sequence_token, pad_sequence, shift_sequence


CODEBOOKS = 4
PADDING_VALUE = get_pad_token_id()


def get_sample_data(B: int, K: int, T: int) -> torch.Tensor:
    sequence = torch.zeros((B, K, T), dtype=torch.int)

    for k in range(K):
        sequence[:, k] = torch.arange(1, T + 1, 1, dtype=torch.int)

    return sequence


def iterate_over_data_loader(data_loader: DataLoader, device: Device) -> int:
    total_items = 0

    for batch in data_loader:
        full_track, stem, labels, sequence_lengths = get_inputs_and_targets(batch, device)
        shape = full_track.shape
        total_items += shape[0]

    return total_items


def test_restore_initial_sequence():
    sequence_length = 100
    full_track = get_sample_data(1, CODEBOOKS, sequence_length)
    processed_full_track = pad_sequence(apply_interleaving(full_track, PADDING_VALUE), 200, PADDING_VALUE)
    sequence = restore_initial_sequence(processed_full_track, sequence_length)

    assert sequence.shape == torch.Size((1, 4, sequence_length))
    assert torch.equal(sequence, full_track), 'Sequences do not match'


def test_get_model_input():

    random.seed(get_random_seed())
    sequence_length = 4096
    full_track = get_sample_data(1, CODEBOOKS, sequence_length)
    stem = get_sample_data(1, CODEBOOKS, sequence_length)
    max_seq_len = 2048
    max_decoder_seq_len = 512
    start = 1309  # random number generated inside the function

    full_track_input, full_track_sequence_length, stem_input, stem_sequence_length, target = get_model_input(
        full_track.clone(),
        stem.clone(),
        sequence_length,
        max_seq_len,
        max_decoder_seq_len,
        CODEBOOKS,
        PADDING_VALUE,
        use_partial_input_sequence=False,
    )

    expected_full_track_input = cut_sequence(apply_interleaving(full_track.clone()[:, :, start: start + max_seq_len], PADDING_VALUE), max_seq_len, cut_left=True)
    expected_stem_input = shift_sequence(stem[:, :, start + max_seq_len + 1: start + max_seq_len + 1 + max_decoder_seq_len], shifts=1)
    expected_stem_input[:, :, 0:1] = get_start_of_sequence_token(CODEBOOKS)
    expected_stem_input = cut_sequence(apply_interleaving(expected_stem_input, PADDING_VALUE), max_decoder_seq_len)

    expected_target = stem[:, :, start + max_seq_len + 1: start + max_seq_len + 1 + max_decoder_seq_len]
    expected_target = cut_sequence(apply_interleaving(expected_target, PADDING_VALUE), max_decoder_seq_len)

    assert full_track_input.shape[-1] == max_seq_len, "Returned full track should be max_seq_len long"
    assert stem_input.shape[-1] == max_decoder_seq_len, "Returned full track should be max_decoder_seq_len long"
    assert target.shape[-1] == max_decoder_seq_len, "Returned target should be max_decoder_seq_len long"
    assert full_track_sequence_length == max_seq_len
    assert stem_sequence_length == max_decoder_seq_len

    assert torch.equal(full_track_input, expected_full_track_input)
    assert torch.equal(stem_input, expected_stem_input)
    assert torch.equal(target, expected_target)


def test_get_model_input_short_sequence():

    random.seed(get_random_seed())
    sequence_length = 128
    full_track = get_sample_data(1, CODEBOOKS, sequence_length)
    stem = get_sample_data(1, CODEBOOKS, sequence_length)
    max_seq_len = 2048
    max_decoder_seq_len = 512

    with pytest.raises(Exception):
        get_model_input(
            full_track.clone(),
            stem.clone(),
            sequence_length,
            max_seq_len,
            max_decoder_seq_len,
            CODEBOOKS,
            PADDING_VALUE,
            use_partial_input_sequence=False,
        )


def test_get_model_input_partial():

    random.seed(get_random_seed())
    sequence_length = 4096
    full_track = get_sample_data(1, CODEBOOKS, sequence_length)
    stem = get_sample_data(1, CODEBOOKS, sequence_length)
    max_seq_len = 2048
    max_decoder_seq_len = 512
    start = 1309  # random number generated inside the function
    end = 1601  # random number generated inside the function
    length = end - start + CODEBOOKS - 1

    full_track_input, full_track_sequence_length, stem_input, stem_sequence_length, target = get_model_input(
        full_track.clone(),
        stem.clone(),
        sequence_length,
        max_seq_len,
        max_decoder_seq_len,
        CODEBOOKS,
        PADDING_VALUE,
        use_partial_input_sequence=True,
    )

    expected_full_track_input = pad_sequence(cut_sequence(apply_interleaving(full_track.clone()[:, :, start:end], PADDING_VALUE), max_seq_len, cut_left=True), max_seq_len, PADDING_VALUE, pad_left=True)
    expected_stem_input = shift_sequence(stem[:, :, end + 1:end + 1 + max_decoder_seq_len], shifts=1)
    expected_stem_input[:, :, 0:1] = get_start_of_sequence_token(CODEBOOKS)
    expected_stem_input = cut_sequence(apply_interleaving(expected_stem_input, PADDING_VALUE), max_decoder_seq_len)

    expected_target = stem[:, :, end + 1:end + 1 + max_decoder_seq_len]
    expected_target = cut_sequence(apply_interleaving(expected_target, PADDING_VALUE), max_decoder_seq_len)

    assert full_track_input.shape[-1] == max_seq_len, "Returned full track should be max_seq_len long"
    assert stem_input.shape[-1] == max_decoder_seq_len, "Returned full track should be max_decoder_seq_len long"
    assert target.shape[-1] == max_decoder_seq_len, "Returned target should be max_decoder_seq_len long"
    assert full_track_sequence_length == length
    assert stem_sequence_length == max_decoder_seq_len

    assert torch.equal(full_track_input, expected_full_track_input)
    assert torch.equal(stem_input, expected_stem_input)
    assert torch.equal(target, expected_target)


def measure_dataloader_iteration_time(data_loader: DataLoader, device: Device):
    print('Measuring access time to dataloader...')
    start_time = time.time()
    total_items = iterate_over_data_loader(data_loader, device)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total time to iterate: {total_time:.2f} seconds")
    print(f"Total number of items: {total_items}")
    print(f"Average time per item: {total_time / total_items:.5f} seconds")
    print(f"Items per second: {total_items / total_time:.2f}")


if __name__ == '__main__':
    data_loader = get_data_loader('train')
    device = get_device()
    measure_dataloader_iteration_time(data_loader, device)