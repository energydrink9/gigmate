import glob
import os
import shutil
from typing import Any, List, Tuple, cast
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from audiomentations import Compose, PitchShift, TimeStretch, Gain
import soundfile

from gigmate.utils.audio_utils import clamp_audio_data, convert_audio_to_int_16

SOURCE_FILES_DIR = '/Users/michele/Music/soundstripe/merged'
OUTPUT_FILES_DIR = '/Users/michele/Music/soundstripe/augmented'
AUGMENTATIONS_COUNT = 3


def get_full_track_files(dir: str):
    return glob.glob(os.path.join(dir, '**/all.ogg'), recursive=True)


def get_stem_files(dir: str):
    return glob.glob(os.path.join(dir, '**/stem.ogg'), recursive=True)


def augment(file_paths: List[Tuple[str, str]], transform: Compose) -> None:

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
        segment.export(output_file_path, format="ogg", codec="libvorbis")


def augment_pitch_and_tempo(file_paths: List[Tuple[str, str]]) -> None:
    transform = Compose(
        transforms=[
            PitchShift(p=1),
            TimeStretch(p=1, leave_length_unchanged=False),
            Gain(p=1, min_gain_db=-9, max_gain_db=9)
        ],
        p=1,
    )

    augment(file_paths, transform)


def augment_all(source_directory: str, output_directory: str):
    files = get_full_track_files(source_directory)
    
    for file_path in tqdm(files, "Augmenting audio tracks"):
        file_dir = os.path.dirname(file_path)
        stem_file_path = os.path.join(file_dir, 'stem.ogg')
        file_dir = os.path.dirname(file_path)
        relative_path = os.path.relpath(file_dir, source_directory)
        output_file_path = os.path.join(output_directory, relative_path + '-original')
        full_track_output_file_path = os.path.join(output_file_path, os.path.basename(file_path))
        stem_output_file_path = os.path.join(output_file_path, os.path.basename(stem_file_path))

        if not os.path.exists(full_track_output_file_path) or not os.path.exists(stem_output_file_path):
            os.makedirs(os.path.dirname(full_track_output_file_path), exist_ok=True)
            shutil.copy(file_path, full_track_output_file_path)
            shutil.copy(stem_file_path, stem_output_file_path)

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
    
    return output_directory


if __name__ == '__main__':
    # random.seed(get_random_seed())
    augment_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR)