import os
import shutil
from typing import Any, List, Tuple, cast
from dask.distributed import Client
from distributed import progress
import numpy as np
from pydub import AudioSegment
from audiomentations import Compose, PitchShift, TimeStretch, Gain
import soundfile
from s3fs.core import S3FileSystem

from gigmate.data_preprocessing.cluster import get_client
from gigmate.utils.audio_utils import clamp_audio_data, convert_audio_to_int_16

BUCKET = 's3://soundstripe-dataset'
SOURCE_FILES_DIR = '/merged'
OUTPUT_FILES_DIR = '/augmented'
AUGMENTATIONS_COUNT = 3

# Set this flag to True to run locally (i.e. not on Coiled)
RUN_LOCALLY = False


def get_full_track_files(fs: S3FileSystem, dir: str) -> List[str]:
    return cast(List[str], fs.glob(os.path.join(dir, '**/all.ogg')))


def augment_files(file_paths: List[Tuple[str, str]], transform: Compose) -> None:

    for file_path, output_file_path in file_paths:
        audio, sr = soundfile.read(file_path, dtype='float32')
        audio = cast(np.ndarray[Any, np.dtype[np.float32]], audio)
        channels = audio.shape[1]
        audio = np.transpose(audio)
        augmented_audio = audio
        augmented_audio = transform(audio, sample_rate=sr)
        transform.freeze_parameters()
        length = augmented_audio.shape[1]
        correct_length = length - (length % (augmented_audio.dtype.itemsize * sr))
        augmented_audio = augmented_audio[:, :correct_length]
        augmented_audio = np.transpose(augmented_audio).reshape(-1)
        augmented_audio = convert_audio_to_int_16(clamp_audio_data(augmented_audio))
        # Using AudioSegment to save to file as soundfile presents a bug with saving in OGG format
        segment = AudioSegment(data=augmented_audio, sample_width=augmented_audio.dtype.itemsize, frame_rate=sr, channels=channels)
        segment.export(output_file_path, format="ogg", codec="libopus")


def augment_pitch_and_tempo(file_paths: List[Tuple[str, str]]) -> None:
    transform = Compose(
        transforms=[
            PitchShift(p=1),
            TimeStretch(p=1, leave_length_unchanged=False),
            Gain(p=1, min_gain_db=-6, max_gain_db=6)
        ],
        p=1,
    )

    augment_files(file_paths, transform)


def augment(params: Tuple[str, str, str]) -> None:
    file_path, source_directory, output_directory = params
    try:
        file_dir = os.path.dirname(file_path)
        stem_file_path = os.path.join(file_dir, 'stem.ogg')
        file_dir = os.path.dirname(file_path)
        relative_path = os.path.relpath(file_dir, source_directory)
        output_file_path = os.path.join(output_directory, relative_path + '-original')
        full_track_output_file_path = os.path.join(output_file_path, os.path.basename(file_path))
        stem_output_file_path = os.path.join(output_file_path, os.path.basename(stem_file_path))

        if not os.path.exists(full_track_output_file_path) or not os.path.exists(stem_output_file_path):
            os.makedirs(os.path.dirname(full_track_output_file_path), exist_ok=True)
            if os.path.exists(file_path):
                shutil.copyfile(file_path, full_track_output_file_path)
            if os.path.exists(stem_file_path):
                shutil.copyfile(stem_file_path, stem_output_file_path)

        for i in range(AUGMENTATIONS_COUNT):
            file_dir = os.path.dirname(file_path)
            relative_path = os.path.relpath(file_dir, source_directory)
            output_file_path = os.path.join(output_directory, relative_path + f'-augmented{i}')
            full_track_output_file_path = os.path.join(output_file_path, os.path.basename(file_path))
            stem_output_file_path = os.path.join(output_file_path, os.path.basename(stem_file_path))
            if not os.path.exists(full_track_output_file_path) or not os.path.exists(stem_output_file_path):
                os.makedirs(output_file_path, exist_ok=True)
                augment_pitch_and_tempo(
                    [
                        (file_path, full_track_output_file_path),
                        (stem_file_path, stem_output_file_path)
                    ]
                )
    except Exception as e:
        print(f'Error augmenting file {file_path}: {e}')


def augment_all(source_directory: str, output_directory: str):

    fs = S3FileSystem()
    files = get_full_track_files(fs, f'{BUCKET}{source_directory}')

    client, dataset_path = cast(Tuple[Client, str], get_client(RUN_LOCALLY, mount_bucket=BUCKET, n_workers=[4, 100]))

    source_directory = dataset_path + source_directory
    output_directory = dataset_path + output_directory
    
    params_list: List[Tuple[str, str, str]] = [(os.path.join('/mount', file_path), source_directory, output_directory) for file_path in files]

    print('Augmenting audio tracks')
    futures = client.map(augment, params_list)
    progress(futures)

    return output_directory


if __name__ == '__main__':
    augment_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR)