

import random
import time
import pytest
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gigmate.dataset.dataset import AudioDataset, get_data_loader, get_dataset, get_inputs_and_targets, get_model_input, restore_initial_sequence
from gigmate.model.model import TransformerModel, get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.utils.device import Device, get_device
from gigmate.utils.constants import get_pad_token_id, get_random_seed, get_start_of_sequence_token_id
from gigmate.utils.sequence_utils import apply_interleaving, apply_start_tokens_to_interleaved_sequence, cut_sequence, get_start_of_sequence_token, pad_sequence, shift_sequence


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
        shift=max_decoder_seq_len,
    )

    expected_full_track_input = cut_sequence(full_track.clone()[:, :, start: start + max_seq_len], max_seq_len, cut_left=True)
    expected_stem_input = shift_sequence(stem[:, :, start + max_seq_len: start + max_seq_len + max_decoder_seq_len], shifts=1)
    expected_stem_input[:, :, 0:1] = get_start_of_sequence_token(CODEBOOKS)
    expected_stem_input = cut_sequence(apply_start_tokens_to_interleaved_sequence(apply_interleaving(expected_stem_input, PADDING_VALUE), 4, get_start_of_sequence_token_id()), max_decoder_seq_len)

    expected_target = stem[:, :, start + max_seq_len: start + max_seq_len + max_decoder_seq_len]
    expected_target = cut_sequence(apply_interleaving(expected_target, PADDING_VALUE), max_decoder_seq_len)

    assert full_track_input.shape[-1] == max_seq_len, "Returned full track should be max_seq_len long"
    assert stem_input.shape[-1] == max_decoder_seq_len, "Returned full track should be max_decoder_seq_len long"
    assert target.shape[-1] == max_decoder_seq_len, "Returned target should be max_decoder_seq_len long"
    assert full_track_sequence_length == max_seq_len
    assert stem_sequence_length == max_decoder_seq_len

    assert torch.equal(full_track_input, expected_full_track_input)
    assert torch.equal(stem_input, expected_stem_input)
    assert torch.equal(target, expected_target)


def test_get_model_input_with_zero_shift():

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
        shift=0,
    )

    expected_full_track_input = cut_sequence(full_track.clone()[:, :, start: start + max_seq_len], max_seq_len, cut_left=True)
    expected_stem_input = shift_sequence(stem[:, :, start + max_seq_len - max_decoder_seq_len: start + max_seq_len], shifts=1)
    expected_stem_input[:, :, 0:1] = get_start_of_sequence_token(CODEBOOKS)
    expected_stem_input = cut_sequence(apply_start_tokens_to_interleaved_sequence(apply_interleaving(expected_stem_input, PADDING_VALUE), 4, get_start_of_sequence_token_id()), max_decoder_seq_len, cut_left=False)

    expected_target = stem[:, :, start + max_seq_len - max_decoder_seq_len: start + max_seq_len]
    expected_target = cut_sequence(apply_interleaving(expected_target, PADDING_VALUE), max_decoder_seq_len, cut_left=False)

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
            shift=max_decoder_seq_len,
        )


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


def compute_embeddings(model: TransformerModel, dataset: AudioDataset, device: str):
    embeddings_list = []
    max_len = 8092
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            item = dataset.__getitem__(idx)
            stem = item.stem.to(device)
            sequence = pad_sequence(cut_sequence(apply_interleaving(stem, get_pad_token_id()), max_len), max_len, get_pad_token_id())
            embeddings = model.compute_embeddings(sequence, model.embeddings)
            embeddings_list.append(embeddings.flatten().cpu())

    return torch.vstack(embeddings_list).numpy()


def detect_outliers(dataset: AudioDataset, device: str, model: TransformerModel):

    def compute_variance(el: torch.Tensor) -> float:
        return el.to(dtype=torch.float).var(dim=-1).sum().item()

    variances = []

    for idx in tqdm(range(len(dataset))):
        item = dataset.__getitem__(idx)
        variance = compute_variance(item.stem)
        variances.append({'variance': variance, 'path': item.path})

    outliers = sorted(variances, key=lambda d: d['variance'], reverse=True)

    for outlier in outliers[:25]:
        print(outlier['path'])


if __name__ == '__main__':
    dataset = get_dataset('test')
    data_loader = get_data_loader('test')
    device = get_device()
    model = get_model(compile=False, checkpoint_path=get_latest_model_checkpoint_path()).to(device)
    detect_outliers(dataset, device, model)
    measure_dataloader_iteration_time(data_loader, device)
