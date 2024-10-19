from dataclasses import dataclass
import glob
import multiprocessing
import pickle
from typing import Dict, List, Tuple, Union
from clearml import Dataset as ClearmlDataset
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import os
import re
import random

from gigmate.utils.constants import get_clearml_dataset_name, get_clearml_dataset_version, get_params, get_clearml_project_name, get_pad_token_id, get_clearml_dataset_tags, get_start_of_sequence_token_id
from gigmate.utils.sequence_utils import apply_interleaving, cut_sequence, pad_sequence, revert_interleaving, shift_sequence

params = get_params()

MIN_TOKENS_TO_KEEP = 64


def get_chunk_number(file_path):
    pattern = r'all-c(\d+)\.pkl$'
    match = re.search(pattern, file_path)
    if match:
        return int(match.group(1))
    return None


def get_stem_file_path(file_path: str) -> str:
    chunk_number = get_chunk_number(file_path)
    assert chunk_number is not None, f'Could not extract chunk number from file path: {file_path}'
    return os.path.join(os.path.dirname(file_path), f'stem-c{chunk_number}.pkl')


def get_entries(dir: str) -> List[Tuple[str, str]]:
    full_tracks_file_paths = glob.glob(os.path.join(dir, '**/all-*.pkl'), recursive=True)

    entries = []

    for file_path in full_tracks_file_paths:
        stem_file_path = get_stem_file_path(file_path)
        if os.path.exists(stem_file_path):
            entries.append((file_path, stem_file_path))
        else:
            print(f'Unable to retrieve stem file: {stem_file_path}')

    return entries


class AudioDataset(Dataset):
    dir: str
    entries: List[Tuple[str, str]]

    def __init__(self, dir: str):
        self.dir = dir
        self.entries = get_entries(dir)

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):

        full_track_file_path, stem_file_path = self.entries[idx]

        with open(full_track_file_path, 'rb') as full_track_file:
            full_track = pickle.load(full_track_file).to('cpu')
            
        with open(stem_file_path, 'rb') as stem_file:
            stem = pickle.load(stem_file).to('cpu')

        # if lengths do not match, cut the sequences to the shortest length
        full_track_length = full_track.shape[-1]
        stem_length = stem.shape[-1]

        if (full_track_length != stem_length):
            length_to_keep = min(full_track_length, stem_length)
            full_track = cut_sequence(full_track, length_to_keep)
            stem = cut_sequence(stem, length_to_keep)

        return full_track, stem
    

def get_dataset(directory: str):
    dataset = AudioDataset(directory)
    return dataset


def get_remote_dataset(dataset_set: str) -> str:
    dataset = ClearmlDataset.get(
        alias=f'{get_clearml_dataset_name()}-{dataset_set}',
        dataset_project=get_clearml_project_name(),
        dataset_name=get_clearml_dataset_name(),
        dataset_version=get_clearml_dataset_version(),
        dataset_tags=[f"{dataset_set}-set", *get_clearml_dataset_tags()],
        only_completed=False,
        only_published=False,
    )
    return dataset.get_local_copy()


def get_pt_dataset_from_remote_dataset(set: str):
    dataset = get_remote_dataset(set)
    return get_dataset(dataset)


def get_datasets():
    train_ds = get_pt_dataset_from_remote_dataset('train')
    validation_ds = get_pt_dataset_from_remote_dataset('validation')
    test_ds = get_pt_dataset_from_remote_dataset('test')
    return train_ds, validation_ds, test_ds


def restore_initial_sequence(tensor: Tensor, sequence_length: int) -> Tensor:
    tensor = revert_interleaving(tensor)
    tensor = cut_sequence(tensor, sequence_length)
    return tensor


@dataclass
class ModelInput():
    full_track: Tensor
    stem: Tensor


@dataclass
class SequenceLengths():
    full_track: List[int]
    stem: List[int]


def decoder_only_collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Dict[str, Union[ModelInput, Tensor, SequenceLengths]]:
    full_tracks: List[Tensor] = []
    stems: List[Tensor] = []
    targets: List[Tensor] = []
    full_track_sequence_lengths = []
    stem_sequence_lengths = []

    padding_value = get_pad_token_id()
    max_seq_len = get_params()['max_seq_len']
    max_decoder_seq_len = get_params()['max_decoder_seq_len']
    
    for full_track, stem in batch:
        assert full_track.ndim == 3, 'Expected 3 dimensions for full track'
        assert stem.ndim == 3, 'Expected 3 dimensions for stem'
        assert full_track.shape == stem.shape, 'Full track and stem have different shapes'

        _, codebooks, sequence_length = full_track.shape

        # We need at least MIN_TOKENS_TO_KEEP tokens both in the full track and in the stem
        if sequence_length < MIN_TOKENS_TO_KEEP * 2:
            full_tracks.append(torch.full((1, codebooks, max_seq_len), get_pad_token_id()))
            stems.append(torch.full((1, codebooks, max_decoder_seq_len), get_pad_token_id()))
            targets.append(torch.full((1, codebooks, max_decoder_seq_len), get_pad_token_id()))
            full_track_sequence_lengths.append(0)
            stem_sequence_lengths.append(0)
            continue

        # Keep a random number of elements *length_to_keep* from the sequence in the full track
        max_tokens_to_keep = sequence_length - MIN_TOKENS_TO_KEEP
        length_to_keep = random.randint(MIN_TOKENS_TO_KEEP, max_tokens_to_keep)
        full_track_input = full_track[:, :, :length_to_keep]

        # Remove 1st *length_to_keep* elements from the target and the stem that do not have to be predicted
        stem_input = stem[:, :, length_to_keep + 1:]
        target = stem_input

        # Shift the stem by 1 position and set the first token to the start of sequence token
        stem_input = shift_sequence(stem_input, shifts=1)
        stem_input[:, :, 0] = get_start_of_sequence_token_id()

        # Calculate sequence lengths
        full_track_sequence_length = min(full_track_input.shape[-1] + codebooks - 1, max_seq_len)
        stem_sequence_length = min(stem_input.shape[-1] + codebooks - 1, max_decoder_seq_len)

        # Cut sequences if necessary
        full_track_input = cut_sequence(full_track_input, max_seq_len - codebooks + 1, cut_left=True)
        stem_input = cut_sequence(stem_input, max_decoder_seq_len - codebooks + 1)
        target = cut_sequence(target, max_decoder_seq_len - codebooks + 1)

        # Apply padding to the sequences
        full_track_input = pad_sequence(full_track_input, max_seq_len, padding_value, pad_left=True)
        stem_input = pad_sequence(stem_input, max_decoder_seq_len, padding_value)
        target = pad_sequence(target, max_decoder_seq_len, padding_value)
        
        # Apply interleaving
        full_track_input = apply_interleaving(full_track_input, padding_value)
        stem_input = apply_interleaving(stem_input, padding_value)
        target = apply_interleaving(target, padding_value)

        # Cut sequences if necessary
        full_track_input = cut_sequence(full_track_input, max_seq_len, cut_left=True)
        stem_input = cut_sequence(stem_input, max_decoder_seq_len)
        target = cut_sequence(target, max_decoder_seq_len)

        assert full_track_input.shape == (1, codebooks, max_seq_len), f"Shape of full track is {full_track_input.shape}"
        assert stem_input.shape == (1, codebooks, max_decoder_seq_len), f"Shape of stem is {stem_input.shape}"
        assert target.shape == (1, codebooks, max_decoder_seq_len), f"Shape of target is {target.shape}"
        assert full_track_sequence_length >= MIN_TOKENS_TO_KEEP, f"Full track is shorter than {MIN_TOKENS_TO_KEEP} tokens"
        assert stem_sequence_length >= MIN_TOKENS_TO_KEEP, f"Stem is shorter than {MIN_TOKENS_TO_KEEP} tokens"

        full_tracks.append(full_track_input)
        stems.append(stem_input)
        targets.append(target)
        full_track_sequence_lengths.append(full_track_sequence_length)
        stem_sequence_lengths.append(stem_sequence_length)

    return {
        'inputs': ModelInput(full_track=torch.cat(full_tracks, dim=0), stem=torch.cat(stems, dim=0)),
        'labels': torch.cat(targets, dim=0),
        'sequence_lengths': SequenceLengths(full_track=full_track_sequence_lengths, stem=stem_sequence_lengths),
    }


def get_data_loader(dataset: str):

    ds = get_pt_dataset_from_remote_dataset(dataset)
    num_workers = multiprocessing.cpu_count()
    prefetch_factor = 4
    shuffle = dataset == 'train'

    return DataLoader(
        ds,
        batch_size=params['batch_size'],
        collate_fn=decoder_only_collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        shuffle=shuffle
    )


def get_data_loaders():
    return get_data_loader('train'), get_data_loader('validation'), get_data_loader('test')


def get_inputs_and_targets(batch, device) -> Tuple[Tensor, Tensor, Tensor, SequenceLengths]:
    return batch['inputs'].full_track.to(device), batch['inputs'].stem.to(device), batch['labels'].to(device), batch['sequence_lengths']

