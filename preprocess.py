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
from constants import get_clearml_dataset_name, get_clearml_dataset_version, get_clearml_project_name, get_params
import shutil
import pickle
import kaggle

DATASET_BASE_DIR = './dataset'
BASE_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean'
SPLIT_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean-split'
AUGMENTED_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean-augmented'
FINAL_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean-final'
ITEMS_PER_FILE = 4096

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

def split_files(source_dir, out_dir, tokenizer):
    miditok.utils.split_files_for_training(
        files_paths=list(Path(Path(source_dir).resolve()).glob('**/*.mid')),
        tokenizer=tokenizer,
        save_dir=Path(out_dir),
        max_seq_len=get_params()['max_seq_len'],
        num_overlap_bars=2,
    )

def augment_dataset(source_path: str, out_path: str):
    miditok.data_augmentation.augment_dataset(
        data_path=Path(source_path),
        pitch_offsets=[-12, 12],
        velocity_offsets=[-4, 4],
        duration_offsets=[-0.5, 0.5],
        out_path=Path(out_path)
    )

def preprocess_midi(midi_data, tokenizer):
    return tokenizer.encode(midi_data)

def preprocess_midi_item(item, tokenizer):
    try:
        artist = get_artist_from_file_path(item)
        tokens = preprocess_midi(item, tokenizer)
        return {'artist': artist, 'tokens': tokens.ids }
    except Exception as e:
        print(f'Error loading midi file: {item}')
        print(e)

def preprocess_midi_dataset(midi_data_list, out_dir: str, tokenizer):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f'{len(midi_data_list)} midi files to process')
    tokens_list = thread_map(lambda item: preprocess_midi_item(item, tokenizer), midi_data_list, max_workers=8)
    items_by_artist = itertools.groupby(tokens_list, lambda item: item['artist'])
    for artist, items in items_by_artist:
        items_per_file = itertools.batched(items, ITEMS_PER_FILE)
        for i, item in enumerate(items_per_file):
            artist_dir = os.path.join(out_dir, artist)
            if not os.path.exists(artist_dir):
                os.makedirs(artist_dir, exist_ok=True)

            path = os.path.join(artist_dir, f'item-{artist}-{i}.pkl')
            with open(path, 'ab') as file:
                pickle.dump(item, file)
        
def get_midi_files(dir: str):
    return glob.glob(os.path.join(dir, '**/*.mid'), recursive=True)

def compress_dataset(dir: str, output_file_path: str) -> None:
    return shutil.make_archive(output_file_path, 'zip', dir)

def upload_dataset(path: str, version: str):
    print('Creating dataset')
    dataset = Dataset.create(
        dataset_name=get_clearml_dataset_name(),
        dataset_project=get_clearml_project_name(), 
        dataset_version=version,
#        output_uri="gs://bucket-name/folder",
    )
    print('Adding files')
    dataset.add_files(path=path)
    print('Uploading')
    dataset.upload(show_progress=True, preview=False)
    print('Finalizing')
    dataset.finalize()


if __name__ == '__main__':
    tokenizer = get_tokenizer()
    download_dataset()
    split_files(BASE_DATASET_DIR, SPLIT_DATASET_DIR, tokenizer)
    augment_dataset(SPLIT_DATASET_DIR, AUGMENTED_DATASET_DIR)
    midi_data_list = get_midi_files(AUGMENTED_DATASET_DIR)
    preprocess_midi_dataset(midi_data_list, FINAL_DATASET_DIR, tokenizer)
    upload_dataset(FINAL_DATASET_DIR, get_clearml_dataset_version())
    