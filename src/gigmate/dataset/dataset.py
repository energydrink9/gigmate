import glob
import multiprocessing
import pickle
from typing import Dict, List, Tuple
from clearml import Dataset as ClearmlDataset
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import os
import re

from gigmate.utils.constants import get_clearml_dataset_training_version, get_params, get_clearml_dataset_training_name, get_clearml_project_name, get_pad_token_id
from gigmate.utils.sequence_utils import apply_interleaving, cut_sequence, pad_sequence, revert_interleaving, shift_sequence

params = get_params()


def get_chunk_number(file_path):
    pattern = r'all-[0-9]*-c(\d+)\.pkl$'
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
    return [(file_path, get_stem_file_path(file_path)) for file_path in full_tracks_file_paths ]


class AudioDataset(Dataset):
    dir: str
    entries: List[Tuple[str, str]]

    def __init__(self, dir: str):
        self.dir = dir
        self.entries = get_entries(dir)

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):

        entry = self.entries[idx]

        with open(entry[0], 'rb') as full_track_file:
            full_track = pickle.load(full_track_file).to('cpu')
            
        with open(entry[1], 'rb') as stem_file:
            stem = pickle.load(stem_file).to('cpu')

        return full_track, stem
    

def get_dataset(directory: str):
    dataset = AudioDataset(directory)
    return dataset


def get_remote_dataset(dataset_set: str) -> str:
    dataset = ClearmlDataset.get(
        alias=f'{get_clearml_dataset_training_name()}-{dataset_set}',
        dataset_project=get_clearml_project_name(),
        dataset_name=get_clearml_dataset_training_name(),
        dataset_version=get_clearml_dataset_training_version(),
        dataset_tags=[f"{dataset_set}-set"],
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


def encoder_decoder_collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Dict[str, Tensor]:
    inputs: List[Tensor] = []
    targets: List[Tensor] = []
    sequence_lengths: List[Tensor] = []

    padding_value = get_pad_token_id()
    max_seq_len = get_params()['max_seq_len']
    
    for full_track, stem in batch:
        assert full_track.ndim == 3, 'Expected 3 dimensions for full track'
        assert stem.ndim == 3, 'Expected 3 dimensions for stem'
        assert full_track.shape == stem.shape, 'Full track and stem have different shapes'

        batch_size, codebooks, sequence_length = full_track.shape

        # Remove last token that does not have label
        full_track = full_track[:, :, :-1]

        # Shift labels by 1 position and remove last token that is the result of shifting
        target = shift_sequence(stem)
        target = target[:, :, :-1]

        # Apply padding to sequences
        full_track_padded = pad_sequence(full_track, max_seq_len - codebooks + 1, padding_value)
        target = pad_sequence(target, max_seq_len - codebooks + 1, padding_value)
        
        # Apply interleaving
        input = apply_interleaving(full_track_padded, padding_value)
        target = apply_interleaving(target, padding_value)

        # Cut sequences if necessary
        input = cut_sequence(input, max_seq_len)
        target = cut_sequence(target, max_seq_len)

        inputs.append(input)
        targets.append(target)

        sequence_lengths.append(torch.tensor([sequence_length - 1])) # sequence length has been reduced by one because of shifting
    
    return { 'inputs': torch.cat(inputs, dim=0), 'labels': torch.cat(targets, dim=0), 'sequence_lengths': torch.cat(sequence_lengths) }


def decoder_only_collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Dict[str, Tensor]:
    inputs: List[Tensor] = []
    targets: List[Tensor] = []
    sequence_lengths: List[Tensor] = []

    padding_value = get_pad_token_id()
    max_seq_len = get_params()['max_seq_len']
    
    for full_track, stem in batch:
        assert full_track.ndim == 3, 'Expected 3 dimensions for full track'
        assert stem.ndim == 3, 'Expected 3 dimensions for stem'
        assert full_track.shape == stem.shape, 'Full track and stem have different shapes'

        batch_size, codebooks, sequence_length = full_track.shape

        # Shift labels by 1 position and remove last token that is the result of shifting
        target = shift_sequence(full_track)
        target = target[:, :, :-1]

        # Remove last token that does not have label
        input = full_track[:, :, :-1]

        # Apply padding to sequences
        input = pad_sequence(input, max_seq_len - codebooks + 1, padding_value)
        target = pad_sequence(target, max_seq_len - codebooks + 1, padding_value)
        
        # Apply interleaving
        input = apply_interleaving(input, padding_value)
        target = apply_interleaving(target, padding_value)

        # Cut sequences if necessary
        input = cut_sequence(input, max_seq_len)
        target = cut_sequence(target, max_seq_len)

        inputs.append(input)
        targets.append(target)
        sequence_lengths.append(torch.tensor([sequence_length - 1]))
    
    return { 'inputs': torch.cat(inputs, dim=0), 'labels': torch.cat(targets, dim=0), 'sequence_lengths': torch.cat(sequence_lengths) }
    # TODO: use nested tensors
    #return { 'inputs': torch.nested.nested_tensor(inputs), 'labels': torch.nested.nested_tensor(targets) }


def get_data_loader(dataset: str):

    ds = get_pt_dataset_from_remote_dataset(dataset)
    num_workers = multiprocessing.cpu_count()
    prefetch_factor = 4

    # The datasets are already shuffled during preprocessing
    return DataLoader(
        ds,
        batch_size=params['batch_size'],
        collate_fn=decoder_only_collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )


def get_data_loaders():
    return get_data_loader('train'), get_data_loader('validation'), get_data_loader('test')


def get_inputs_and_targets(batch, device):
    return batch['inputs'].to(device), batch['labels'].to(device), batch['sequence_lengths'].to(device)

