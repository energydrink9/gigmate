import glob
import os
from zipfile import ZipFile
from tqdm import tqdm

COMPRESSED_FILES_DIR = '/Users/michele/Music/soundstripe/original'

def get_compressed_files(dir: str):
    return glob.glob(os.path.join(dir, '**/*.zip'), recursive=True)

def uncompress_files(directory: str) -> str:
    files = get_compressed_files(directory)
    
    for filename in tqdm(files, "Uncompressing files"):
        with ZipFile(filename, 'r') as zip_file:
            zip_file.extractall(os.path.dirname(filename))
        os.remove(filename)

    return directory

if __name__ == '__main__':
    uncompress_files(COMPRESSED_FILES_DIR)