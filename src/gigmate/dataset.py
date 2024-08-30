import multiprocessing
from pathlib import Path, PurePath
from clearml import Dataset
import torch
from torch.utils.data import DataLoader
from miditok.pytorch_data import DataCollator
import os
from sklearn.model_selection import train_test_split
from gigmate.constants import get_clearml_dataset_version, get_params, get_random_seed, get_clearml_dataset_name, get_clearml_project_name, get_pad_token_id
from gigmate.DatasetPickle import DatasetPickle
import time
import random

random.seed(get_random_seed())
torch.manual_seed(get_random_seed())

params = get_params()

def get_parent_directory_name_from_file_path(file_path):
    path = PurePath(file_path)
    return path.parent.name

def get_filename_from_file_path(file_path):
    return os.path.basename(file_path)

# Function to generate file paths
def get_file_paths(directory):
    return list(Path(directory).glob('**/*.pkl'))

def split_by_artist(file_paths_with_artists, artists, validation_size, test_size, seed=get_random_seed()):
    train_artists, rest_artists = train_test_split(artists, test_size=validation_size + test_size, random_state=seed)
    validation_artists, test_artists = train_test_split(rest_artists, test_size=0.5, random_state=seed)

    train_paths = [path for path, artist in file_paths_with_artists if artist in train_artists]
    validation_paths = [path for path, artist in file_paths_with_artists if artist in validation_artists]
    test_paths = [path for path, artist in file_paths_with_artists if artist in test_artists]

    return train_paths, validation_paths, test_paths

def get_dataset(directory: str, max_seq_len: int):
    dataset = DatasetPickle(
        directory=directory,
        max_seq_len=max_seq_len,
        bos_token_id=1,
        eos_token_id=2,
    )
    return dataset

def create_pt_dataset(directory, max_seq_len):
    return get_dataset(directory, max_seq_len)

def get_remote_dataset(dataset: str):
    dataset = Dataset.get(
        alias=f'{get_clearml_dataset_name()}-{dataset}',
        dataset_project=get_clearml_project_name(),
        dataset_name=get_clearml_dataset_name(),
        dataset_version=get_clearml_dataset_version(),
        dataset_tags=[f"{dataset}-set"],
        only_completed=False, 
        only_published=False, 
    )
    return dataset.get_local_copy()

def get_pt_dataset_from_remote_dataset(set: str):
    dataset = get_remote_dataset(set)
    return create_pt_dataset(dataset, max_seq_len=get_params()['max_seq_len'])

def get_datasets():
    train_ds = get_pt_dataset_from_remote_dataset('train')
    validation_ds = get_pt_dataset_from_remote_dataset('validation')
    test_ds = get_pt_dataset_from_remote_dataset('test')
    return train_ds, validation_ds, test_ds

def get_data_loaders():

    train_ds, validation_ds, test_ds = get_datasets()

    collator = DataCollator(
        pad_token_id=get_pad_token_id(),
        copy_inputs_as_labels=True,
        shift_labels=True,
        labels_pad_idx=get_pad_token_id(),
    )

    num_workers = multiprocessing.cpu_count()
    prefetch_factor = 4

    # The train dataset is already shuffled
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], collate_fn=collator, pin_memory=True, num_workers=num_workers, persistent_workers=True, prefetch_factor=prefetch_factor)
    validation_loader = DataLoader(validation_ds, batch_size=params['batch_size'], collate_fn=collator, pin_memory=True, num_workers=num_workers, persistent_workers=True, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_ds, batch_size=params['batch_size'], collate_fn=collator, pin_memory=True, num_workers=num_workers, persistent_workers=True, prefetch_factor=prefetch_factor)

    return train_loader, validation_loader, test_loader

def measure_dataloader_iteration_time(dataloader):
    print('Measuring access time to dataloader...')
    start_time = time.time()
    total_items = 0

    for batch in dataloader:
        # Simulate a light operation on the batch
        shape = batch['input_ids'].shape
        total_items += shape[0]

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total time to iterate: {total_time:.2f} seconds")
    print(f"Total number of items: {total_items}")
    print(f"Average time per item: {total_time / total_items:.5f} seconds")
    print(f"Items per second: {total_items / total_time:.2f}")

if __name__ == '__main__':
    train_dataloader, _, _ = get_data_loaders()
    measure_dataloader_iteration_time(train_dataloader)