import os
from zipfile import ZipFile
import os
from tqdm.contrib.concurrent import thread_map
import miditok
from pathlib import Path, PurePath
import glob
from tokenizer import get_tokenizer
from constants import get_params

DATASET_BASE_DIR = './dataset'
BASE_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean'
SPLIT_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean-split'
AUGMENTED_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean-augmented'
JSON_DATASET_DIR = f'{DATASET_BASE_DIR}/lakh-midi-clean-json'

# # Set up Kaggle credentials
# os.environ['KAGGLE_USERNAME'] = 'michelelugano'
# os.environ['KAGGLE_KEY'] = '4cc37f942c85af673a34905c5466ef23'

# !kaggle datasets download -d imsparsh/lakh-midi-clean
# !pip install kaggle
# !pip install zipfile

# # Set up Kaggle credentials
# os.environ['KAGGLE_USERNAME'] = 'michelelugano'
# os.environ['KAGGLE_KEY'] = '4cc37f942c85af673a34905c5466ef23'

# !kaggle datasets download -d imsparsh/lakh-midi-clean

# # Unzip the dataset
# with ZipFile('lakh-midi-clean.zip', 'r') as zip_ref:
#     zip_ref.extractall('lakh-midi-clean')

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

def preprocess_midi(midi_data):
    return tokenizer(midi_data)

def preprocess_midi_item(item, out_dir):
    (i, midi_path) = item
    try:
        artist = get_artist_from_file_path(midi_path)
        filename = f'{artist}-{get_filename_from_file_path(midi_path)}'
        file_path = get_json_file_path(out_dir, artist, filename)
        dir = get_json_file_dir(out_dir, artist)
        if not os.path.exists(file_path):
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)
            tokens = preprocess_midi(midi_path)
            tokenizer.save_tokens(tokens, path=file_path)
    except Exception as e:
        print(f'Error loading midi file: {midi_path}')
        print(e)

def preprocess_midi_dataset(midi_data_list, out_dir: str):
    print(f'{len(midi_data_list)} midi files to process')
    thread_map(lambda item: preprocess_midi_item(item, out_dir), enumerate(midi_data_list), max_workers=8)

def get_midi_files(dir: str):
    return glob.glob(os.path.join(dir, '**/*.mid'), recursive=True)


if __name__ == '__main__':
    tokenizer = get_tokenizer()
    extract_dataset('lakh-midi-clean.zip', 'lakh-midi-clean')
    split_files(BASE_DATASET_DIR, SPLIT_DATASET_DIR)
    augment_dataset(SPLIT_DATASET_DIR, AUGMENTED_DATASET_DIR)
    midi_data_list = get_midi_files(AUGMENTED_DATASET_DIR)
    preprocess_midi_dataset(midi_data_list, JSON_DATASET_DIR)