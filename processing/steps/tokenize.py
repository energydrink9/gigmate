from clearml import Task
import os
import glob
from tqdm.contrib.concurrent import thread_map
import multiprocessing
import itertools
import pickle
import json
from sklearn.model_selection import train_test_split
from pathlib import PurePath
from gigmate.constants import get_clearml_dataset_version, get_clearml_project_name, get_random_seed
from gigmate.tokenizer import get_tokenizer
from gigmate.processing.process import upload_dataset

FINAL_DATASET_DIR = './dataset/lakh-midi-clean-final'
ITEMS_PER_FILE = 1024 * 2

def get_midi_files(dir: str):
    return glob.glob(os.path.join(dir, '**/*.mid'), recursive=True)

def preprocess_midi(midi_file_path, tokenizer):
    return tokenizer.encode(midi_file_path)

def preprocess_midi_items(out_dir, items, tokenizer, i):
    tokenized_items = []
    print(f'Processing {len(items)} items in batch {i}')
    for item in items:
        try:
            tokens = preprocess_midi(item, tokenizer)
            tokenized_items.append(tokens.ids)
        except Exception as e:
            print(f'Error loading midi file: {item}')
            print(e)

    if len(tokenized_items) > 0:
        persist_items_batch(out_dir, i, tokenized_items)
    
    return len(tokenized_items)

def persist_items_batch(out_dir: str, i: int, batch):
    path = os.path.join(out_dir, f'item-{i}.pkl')
    with open(path, 'wb') as file:
        pickle.dump(batch, file)

def preprocess_midi_dataset(midi_data_list, out_dir: str, tokenizer):
    print('Converting to pickle files')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f'{len(midi_data_list)} midi files to process')
    items_batches = list(itertools.batched(midi_data_list, ITEMS_PER_FILE))
    result = thread_map(lambda items: preprocess_midi_items(out_dir, items[1], tokenizer, items[0]), enumerate(items_batches), max_workers=multiprocessing.cpu_count())
    total_files = sum(result)

    metadata = {
        'total_files': total_files
    }
    with open(os.path.join(out_dir, 'metadata'), 'w+') as file:
        json.dump(metadata, file)

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

def tokenize_midi_files(augmented_output_dir: str):
    tokenizer = get_tokenizer()
    midi_data_list = get_midi_files(augmented_output_dir)
    paths_by_dataset = split_dataset(midi_data_list)

    for dataset, paths in paths_by_dataset.items():
        print(f'Processing {dataset} dataset paths')
        directory = os.path.join(FINAL_DATASET_DIR, dataset)
        preprocess_midi_dataset(paths, directory, tokenizer)
        upload_dataset(directory, get_clearml_dataset_version(), ["step-tokenize"], dataset=dataset)

    return FINAL_DATASET_DIR

if __name__ == '__main__':
    Task.init(project_name=get_clearml_project_name(), task_name='tokenize-midi-files')
    tokenize_midi_files()
