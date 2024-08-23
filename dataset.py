import multiprocessing
from pathlib import Path, PurePath
from clearml import Dataset
import torch
from torch.utils.data import DataLoader
from miditok.pytorch_data import DatasetJSON, DataCollator
import os
from sklearn.model_selection import train_test_split
from constants import get_clearml_dataset_version, get_params, get_random_seed, get_clearml_dataset_name, get_clearml_project_name, get_pad_token_id
import shutil

#!clearml-data create --project GigMate --name LakhMidiCleanDatasetFull
#!clearml-data add --id 1a4115ff408e46208e8beabb197d6bde --files dataset/lakh-midi-clean
#!clearml-data close

torch.manual_seed(get_random_seed())
torch.use_deterministic_algorithms(True)

params = get_params()

def get_parent_directory_name_from_file_path(file_path):
    path = PurePath(file_path)
    return path.parent.name

def get_filename_from_file_path(file_path):
    return os.path.basename(file_path)

# Function to generate file paths
def get_file_paths(directory):
    return list(Path(directory).glob('**/*.json'))

def split_by_artist(file_paths_with_artists, artists, validation_size, test_size, seed=get_random_seed()):
    train_artists, rest_artists = train_test_split(artists, test_size=validation_size + test_size, random_state=seed)
    validation_artists, test_artists = train_test_split(rest_artists, test_size=0.5, random_state=seed)

    train_paths = [path for path, artist in file_paths_with_artists if artist in train_artists]
    validation_paths = [path for path, artist in file_paths_with_artists if artist in validation_artists]
    test_paths = [path for path, artist in file_paths_with_artists if artist in test_artists]

    return train_paths, validation_paths, test_paths

# Function to create TensorFlow Dataset from pickled files
def create_pt_datasets(directory, max_seq_len, validation_size=0.1, test_size=0.1):
    file_paths = get_file_paths(directory)

    artists = list(set(map(lambda path: get_parent_directory_name_from_file_path(path), file_paths)))
    file_paths_with_artists = list(map(lambda path: (path, get_parent_directory_name_from_file_path(path)), file_paths))
    train_paths, validation_paths, test_paths = split_by_artist(file_paths_with_artists, artists, validation_size, test_size)

    train_ds = DatasetJSON(
        files_paths = train_paths,
        max_seq_len=max_seq_len,
        bos_token_id=1,
        eos_token_id=2,
    )
    validation_ds = DatasetJSON(
        files_paths = validation_paths,
        max_seq_len=max_seq_len,
        bos_token_id=1,
        eos_token_id=2,
    )
    test_ds = DatasetJSON(
        files_paths = test_paths,
        max_seq_len=max_seq_len,
        bos_token_id=1,
        eos_token_id=2,
    )

    return train_ds, validation_ds, test_ds

def get_remote_dataset():

    dataset = Dataset.get(
        dataset_project=get_clearml_project_name(),
        dataset_name=get_clearml_dataset_name(),
        dataset_version=get_clearml_dataset_version(),
#        dataset_tags="full",
        only_completed=False, 
        only_published=False, 
    )
    return dataset.get_local_copy()

def get_dataset():
    remote_dataset = get_remote_dataset()
    directory = remote_dataset
    train_ds, validation_ds, test_ds = create_pt_datasets(directory, max_seq_len=get_params()['max_seq_len'])
    return train_ds, validation_ds, test_ds

def get_data_loaders():

    train_ds, validation_ds, test_ds = get_dataset()

    collator = DataCollator(
        pad_token_id=get_pad_token_id(),
        copy_inputs_as_labels=True,
        shift_labels=True,
        labels_pad_idx=get_pad_token_id(),
    )

    num_workers = multiprocessing.cpu_count() - 1

    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], collate_fn=collator, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_ds, batch_size=params['batch_size'], collate_fn=collator, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=params['batch_size'], collate_fn=collator, shuffle=False, num_workers=num_workers)

    return train_loader, validation_loader, test_loader