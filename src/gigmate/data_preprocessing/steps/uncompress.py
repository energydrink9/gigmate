import glob
import os
from zipfile import ZipFile
from tqdm import tqdm

ORIGINAL_FILES_DIR = os.path.join('../dataset/original')


def get_compressed_files(dir: str):
    return glob.glob(os.path.join(dir, '**/*.zip'), recursive=True)


def uncompress_files(directory: str) -> str:
    files = get_compressed_files(directory)
    
    for filename in tqdm(files, "Uncompressing files"):
        
        try:
            with ZipFile(filename, 'r') as zip_file:
                zip_file.extractall(os.path.dirname(filename))
            os.remove(filename)

        except Exception as e:
            print(f'Unable to uncompress file: {filename}')
            print(e)

    return directory


if __name__ == '__main__':
    uncompress_files(ORIGINAL_FILES_DIR)