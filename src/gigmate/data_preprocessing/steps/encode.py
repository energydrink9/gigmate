import glob
import os
import pickle
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from gigmate.model.codec import encode_file
from gigmate.utils.device import get_device

SOURCE_FILES_DIR = '/Users/michele/Music/soundstripe/augmented'
OUTPUT_FILES_DIR = '/Users/michele/Music/soundstripe/encoded'

def get_ogg_files(dir: str):
    return glob.glob(os.path.join(dir, '**/*.ogg'), recursive=True)


def encode_all(source_directory: str, output_directory: str):
    files = get_ogg_files(source_directory)
    
    for filename in tqdm(files, "Encoding audio tracks"):
        file_dir = os.path.dirname(filename)
        encoded_chunks, frame_rate = encode_file(filename, 'cpu', add_start_and_end_tokens=True)
        relative_path = os.path.relpath(file_dir, source_directory)
        file_output_directory = os.path.join(output_directory, relative_path)
        os.makedirs(file_output_directory, exist_ok=True)

        for i, chunk in enumerate(encoded_chunks):
            output_filename = os.path.basename(filename).split('.')[0] + f'-c{i}.pkl'
            pickle.dump(chunk.to('cpu'), open(os.path.join(file_output_directory, output_filename), 'wb'))
    
    return source_directory

if __name__ == '__main__':
    encode_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR)