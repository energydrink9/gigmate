import itertools
import os
from zipfile import ZipFile
import os
from clearml import Dataset
from tqdm.contrib.concurrent import thread_map
import miditok
from pathlib import Path, PurePath
import glob
from tokenizer import get_tokenizer
from constants import get_clearml_dataset_name, get_clearml_dataset_version, get_clearml_project_name, get_params, get_random_seed
import shutil
import pickle
import kaggle
from sklearn.model_selection import train_test_split
import multiprocessing
import json
from miditok.utils import get_average_num_tokens_per_note
import random

DATASET_BASE_DIR = './dataset'
BASE_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean'
SPLIT_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean-split'
AUGMENTED_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean-augmented'
FINAL_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean-final'
ITEMS_PER_FILE = 1024 * 8
NUMBER_OF_FILES_TO_COMPUTE_AVERAGE_NUM_TOKENS_PER_NOTE = 100

# # Unzip the dataset
# with ZipFile('lakh-midi-clean.zip', 'r') as zip_ref:
#     zip_ref.extractall('lakh-midi-clean')

def download_dataset():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('imsparsh/lakh-midi-clean', path=BASE_DATASET_DIR, unzip=True)

def get_json_file_dir(directory, artist):
    return f'{directory}/{artist}'

def get_json_file_path(directory: str, artist, filename):
    return f'{get_json_file_dir(directory, artist)}/{filename}.json'

def get_artist_from_file_path(file_path):
    path = PurePath(file_path)
    return path.parent.name

def get_filename_from_file_path(file_path):
    return os.path.basename(file_path)


def extract_dataset(filename: str, out: str):
    with ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(out)

def get_files_in_directory(directory: str, extension: str = '.mid'):
    absolute_path = Path(directory).resolve()
    iterator = Path(absolute_path).glob(f'**/*{extension}')
    return [p for p in iterator]

def directory_has_files(directory: str, file_extension: str):
    return len(get_files_in_directory(directory, file_extension)) > 0

def split_directory_files(directory, out_path, tokenizer, average_num_tokens):
    try:
        miditok.utils.split_files_for_training(
            files_paths=get_files_in_directory(directory),
            tokenizer=tokenizer,
            save_dir=Path(out_path),
            max_seq_len=get_params()['max_seq_len'],
            num_overlap_bars=2,
            average_num_tokens_per_note=average_num_tokens
        )
    except Exception as e:
        print(f'Error splitting files in directory: {directory}')
        raise e

def get_average_num_tokens(directory: str):
    files = get_files_in_directory(directory)
    print(files[0])
    print(directory)
    print(len(files))
    random_files = random.sample(files, min(NUMBER_OF_FILES_TO_COMPUTE_AVERAGE_NUM_TOKENS_PER_NOTE, len(files)))
    return get_average_num_tokens_per_note(tokenizer, random_files)

def split_files(source_path, out_path, tokenizer):
    average_num_tokens = get_average_num_tokens(source_path)
    print(f'Average number of tokens per note: {average_num_tokens}')
    directories = [os.path.basename(f.path) for f in os.scandir(source_path) if f.is_dir() and directory_has_files(f.path, '.mid')]
    thread_map(lambda directory: split_directory_files(os.path.join(source_path, directory), os.path.join(out_path, directory), tokenizer, average_num_tokens), directories, max_workers=multiprocessing.cpu_count())
    print('Data splitting complete')

def augment_dataset_directory(directory: str, out_path: str):
    try:
        miditok.data_augmentation.augment_dataset(
            data_path=Path(directory),
            pitch_offsets=[-10, 10],
            velocity_offsets=[-4, 4],
            duration_offsets=[-0.5, 0.5],
            out_path=Path(out_path)
        )
    except Exception as e:
        print(f'Error augmenting files in directory: {directory}')
        raise e

def augment_dataset(source_path: str, out_path: str):
    directories = [os.path.basename(f.path) for f in os.scandir(source_path) if f.is_dir() and directory_has_files(f.path, '.mid')]
    thread_map(lambda directory: augment_dataset_directory(os.path.join(source_path, directory), os.path.join(out_path, directory)), directories, max_workers=multiprocessing.cpu_count())
    print('Data augmentation complete')

def preprocess_midi(midi_data, tokenizer):
    return tokenizer.encode(midi_data)

def preprocess_midi_item(item, tokenizer):
    try:
        tokens = preprocess_midi(item, tokenizer)
        return tokens.ids
    except Exception as e:
        print(f'Error loading midi file: {item}')
        print(e)

def persist_items_batch(out_dir: str, i: int, item):
    path = os.path.join(out_dir, f'item-{i}.pkl')
    with open(path, 'ab') as file:
        pickle.dump(item, file)

def preprocess_midi_dataset(midi_data_list, out_dir: str, tokenizer):
    print('Converting to pickle files')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f'{len(midi_data_list)} midi files to process')
    items = thread_map(lambda item: preprocess_midi_item(item, tokenizer), midi_data_list, max_workers=multiprocessing.cpu_count())
    items_per_file = itertools.batched(items, ITEMS_PER_FILE)
    thread_map(lambda item: persist_items_batch(out_dir, item[0], item[1]), enumerate(items_per_file), max_workers=multiprocessing.cpu_count())
    
    metadata = {
        'total_files': len(items)
    }
    with open(os.path.join(out_dir, 'metadata'), 'w+') as file:
        json.dump(metadata, file)
        
def get_midi_files(dir: str):
    return glob.glob(os.path.join(dir, '**/*.mid'), recursive=True)

def compress_dataset(dir: str, output_file_path: str) -> None:
    return shutil.make_archive(output_file_path, 'zip', dir)

def upload_dataset(dataset: str, path: str, version: str):
    print('Creating dataset')
    dataset = Dataset.create(
        dataset_name=get_clearml_dataset_name(),
        dataset_project=get_clearml_project_name(), 
        dataset_version=version,
        dataset_tags=[f'{dataset}-set']
#        output_uri="gs://bucket-name/folder",
    )
    print('Adding files')
    dataset.add_files(path=path)
    print('Uploading')
    dataset.upload(show_progress=True, preview=False)
    print('Finalizing')
    dataset.finalize()

def get_parent_directory_name_from_file_path(file_path):
    path = PurePath(file_path)
    return path.parent.name

def split_by_artist(file_paths_with_artists, artists, validation_size, test_size, seed=get_random_seed()):
    train_artists, rest_artists = train_test_split(artists, test_size=validation_size + test_size, random_state=seed)
    validation_artists, test_artists = train_test_split(rest_artists, test_size=0.5, random_state=seed)

    train_paths = [path for path, artist in file_paths_with_artists if artist in train_artists]
    validation_paths = [path for path, artist in file_paths_with_artists if artist in validation_artists]
    test_paths = [path for path, artist in file_paths_with_artists if artist in test_artists]

    return train_paths, validation_paths, test_paths

def split_dataset(file_paths, validation_size=0.1, test_size=0.1):
    artists = list(set(map(lambda path: get_parent_directory_name_from_file_path(path), file_paths)))
    file_paths_with_artists = list(map(lambda path: (path, get_parent_directory_name_from_file_path(path)), file_paths))
    train_paths, validation_paths, test_paths = split_by_artist(file_paths_with_artists, artists, validation_size, test_size)
    return {
        'train': train_paths,
        'validation': validation_paths,
        'test': test_paths
    }

if __name__ == '__main__':
    tokenizer = get_tokenizer()
    download_dataset()
    split_files(BASE_DATASET_DIR, SPLIT_DATASET_DIR, tokenizer)
    augment_dataset(SPLIT_DATASET_DIR, AUGMENTED_DATASET_DIR)
    midi_data_list = get_midi_files(AUGMENTED_DATASET_DIR)
    paths_by_dataset = split_dataset(midi_data_list)

    for dataset, paths in paths_by_dataset.items():
        print(f'Processing {dataset} dataset paths')
        directory = os.path.join(FINAL_DATASET_DIR, dataset)
        preprocess_midi_dataset(paths, directory, tokenizer)
        upload_dataset(dataset, directory, get_clearml_dataset_version())
    