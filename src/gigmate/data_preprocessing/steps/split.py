import glob
import os
import random
import shutil
from typing import List, Set, cast
from s3fs.core import S3FileSystem
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gigmate.utils.constants import get_random_seed

SOURCE_FILES_DIR = '../dataset/encoded'
OUTPUT_FILES_DIR = '../dataset/split'
SPLIT_NAMES = ['train', 'validation', 'test']
VALIDATION_SIZE = 0.8
TEST_SIZE = 0.6


def get_directories_containing_pkl_files(fs: S3FileSystem, dir: str) -> Set[str]:
    files = cast(List[str], fs.glob(os.path.join(dir, '**/*.pkl')))
    directories = {os.path.dirname(file) for file in files}
    
    return directories


def split_by_artist(file_paths_with_artists, artists, validation_size, test_size, seed=get_random_seed()):
    train_artists, rest_artists = train_test_split(artists, test_size=validation_size + test_size, random_state=seed)
    validation_artists, test_artists = train_test_split(rest_artists, test_size=0.5, random_state=seed)

    train_paths = [path for path, artist in file_paths_with_artists if artist in train_artists]
    validation_paths = [path for path, artist in file_paths_with_artists if artist in validation_artists]
    test_paths = [path for path, artist in file_paths_with_artists if artist in test_artists]

    return train_paths, validation_paths, test_paths


def split_all(source_directory: str, output_directory: str) -> List[str]:
    
    fs = S3FileSystem(use_listings_cache=False)

    file_paths = get_directories_containing_pkl_files(fs, source_directory)
    file_paths_artists = [os.path.split(os.path.split(file_path)[0])[-1] for file_path in file_paths]

    file_paths_with_artists = list(zip(file_paths, file_paths_artists))
    artists = list(set(file_paths_artists))
    splits = split_by_artist(file_paths_with_artists, artists, validation_size=VALIDATION_SIZE, test_size=TEST_SIZE)

    output_directories = []

    for i, split in enumerate(splits):
        split_directory = os.path.join(output_directory, SPLIT_NAMES[i])
        output_directories.append(split_directory)

        for split_file in tqdm(split, f'Creating split {SPLIT_NAMES[i]}'):
            relative_path = os.path.relpath(split_file, source_directory)
            
            file_output_dir = os.path.join(split_directory, relative_path)
            shutil.copytree(split_file, file_output_dir, dirs_exist_ok=True)

    return output_directories


if __name__ == '__main__':
    random.seed(get_random_seed())
    split_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR)