import tempfile
from typing import List, cast, Tuple
import os
from s3fs.core import S3FileSystem
from tqdm import tqdm
import multiprocessing
import multiprocessing.pool

from gigmate.data_preprocessing.constants import CLEARML_DATASET_TRAINING_VERSION, DATASET_TAGS
from gigmate.data_preprocessing.utils import upload_dataset

BUCKET_NAME = 'soundstripe-dataset'
PROTOCOL = 's3://'
BUCKET = f'{PROTOCOL}{BUCKET_NAME}'
SOURCE_FILES_DIRS = ['split/train', 'split/validation', 'split/test']


def get_files(fs: S3FileSystem, dir: str) -> List[str]:
    return [PROTOCOL + path for path in cast(List[str], fs.glob(os.path.join(dir, '**/*.pkl')))]


def download_file(params: Tuple[S3FileSystem, str, str, str]):
    (fs, file, source_directory, output_directory) = params
    local_file_path = os.path.join(output_directory, os.path.relpath(file, source_directory))
    if not os.path.exists(local_file_path):
        try:
            fs.get(file, local_file_path, recursive=True)
        except Exception as e:
            print('Error getting the file, retrying')
            print(e)
            try:
                fs.get(file, local_file_path, recursive=True)
            except Exception as e2:
                print('Error getting the file, retrying')
                print(e2)
                try:
                    fs.get(file, local_file_path, recursive=True)
                except Exception as e3:
                    print('Error getting the file, abandoning')
                    print(e3)


def upload(input_dirs: List[str], tags: List[str]):
    fs = S3FileSystem(use_listings_cache=False)

    for split_dir in input_dirs:
        source_directory = os.path.join(BUCKET, split_dir)
        set = os.path.split(source_directory)[1]
        
        with tempfile.TemporaryDirectory() as local_directory:
            print(f'Downloading {set} dataset (folder {source_directory}) from S3 into {local_directory}')
            
            files = get_files(fs, source_directory)
            inputs = [(fs, file, source_directory, local_directory) for file in files]

            with multiprocessing.pool.ThreadPool(multiprocessing.cpu_count()) as pool:
                list(tqdm(pool.imap(download_file, inputs), total=len(inputs)))

            print(f'Uploading {set} dataset to ClearML')
            upload_dataset(path=local_directory, version=CLEARML_DATASET_TRAINING_VERSION, tags=tags + ['final'], dataset_set=set)


if __name__ == '__main__':
    upload(SOURCE_FILES_DIRS, DATASET_TAGS)