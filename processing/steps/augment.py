from pathlib import Path
import os
from tqdm.contrib.concurrent import thread_map
import multiprocessing
from miditok import data_augmentation
from gigmate.processing.process import directory_has_files
from gigmate.processing.steps.split import SPLIT_DATASET_DIR

AUGMENTED_DATASET_DIR = './gigmate/dataset/lakh-midi-clean-augmented'

def augment_dataset_directory(directory: str, out_path: str):
    try:
        data_augmentation.augment_dataset(
            data_path=Path(directory),
            pitch_offsets=[-5, 5],
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
    return out_path

def augment_midi_files(split_output_dir: str): 
    augment_output_dir = augment_dataset(split_output_dir, AUGMENTED_DATASET_DIR)
    return augment_output_dir

if __name__ == '__main__':
    augment_midi_files(SPLIT_DATASET_DIR)