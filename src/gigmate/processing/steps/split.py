from pathlib import Path
import miditok
from gigmate.constants import get_params
from gigmate.tokenizer import get_tokenizer
from gigmate.processing.process import get_files_in_directory
import random
import kaggle
import os

BASE_DATASET_DIR = 'dataset/lakh-midi-clean'
SPLIT_DATASET_DIR = 'dataset/lakh-midi-clean-split'
NUMBER_OF_FILES_TO_COMPUTE_AVERAGE_NUM_TOKENS_PER_NOTE = 100
SEQUENCE_LENGTH = 128

def download_raw_dataset(directory: str):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('imsparsh/lakh-midi-clean', path=directory, unzip=True)
    return directory

def get_average_num_tokens(directory: str, tokenizer):
    files = get_files_in_directory(directory)
    random_files = random.sample(files, min(NUMBER_OF_FILES_TO_COMPUTE_AVERAGE_NUM_TOKENS_PER_NOTE, len(files)))
    return miditok.utils.get_average_num_tokens_per_note(tokenizer, random_files)

def split_directory_files(directory, out_path, tokenizer, average_num_tokens):
    try:
        miditok.utils.split_files_for_training(
            files_paths=get_files_in_directory(directory),
            tokenizer=tokenizer,
            save_dir=Path(out_path),
            max_seq_len=SEQUENCE_LENGTH,
            num_overlap_bars=2,
            average_num_tokens_per_note=average_num_tokens
        )
    except FileNotFoundError as e:
        print(f'Warning: Hidden file not found in {out_path}. Continuing without it.')
    except Exception as e:
        print(f'Error splitting files in directory: {directory}')
        raise e

def split_files(source_path, out_path, tokenizer):
    average_num_tokens = get_average_num_tokens(source_path, tokenizer)
    print(f'Average number of tokens per note: {average_num_tokens}')
    split_directory_files(source_path, out_path, tokenizer, average_num_tokens)
    print('Data splitting complete')
    return SPLIT_DATASET_DIR

def split_midi_files():
    if not os.path.exists(BASE_DATASET_DIR):
        os.makedirs(BASE_DATASET_DIR, exist_ok=True)

    raw_dataset_dir = download_raw_dataset(BASE_DATASET_DIR)
    tokenizer = get_tokenizer()
    split_files(raw_dataset_dir, SPLIT_DATASET_DIR, tokenizer)
    return SPLIT_DATASET_DIR

if __name__ == '__main__':
    split_midi_files()
