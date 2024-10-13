import glob
import os
import shutil
from typing import List, Tuple
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from audiomentations import Compose, PitchShift, TimeStretch

from gigmate.utils.audio_utils import clamp_audio_data, convert_audio_to_float_32, convert_audio_to_int_16

SOURCE_FILES_DIR = '/Users/michele/Music/soundstripe/merged'
OUTPUT_FILES_DIR = '/Users/michele/Music/soundstripe/augmented'
TEMPO_AUGMENTATIONS_COUNT = 4
PITCH_AUGMENTATIONS_COUNT = 4


def get_full_track_files(dir: str):
    return glob.glob(os.path.join(dir, '**/all-*.ogg'), recursive=True)


def get_stem_files(dir: str):
    return glob.glob(os.path.join(dir, '**/stem.ogg'), recursive=True)


def augment(audio_segments: List[AudioSegment], transform: Compose) -> List[AudioSegment]:
    augmented_audio_segments = []
    for audio_segment in audio_segments:
        sample_rate = audio_segment.frame_rate
        channels = audio_segment.channels

        audio = convert_audio_to_float_32(np.array(audio_segment.get_array_of_samples()))
        augmented_audio = transform(audio, sample_rate=sample_rate)
        transform.freeze_parameters()
        augmented_audio = convert_audio_to_int_16(clamp_audio_data(augmented_audio))
        augmented_audio = augmented_audio.reshape((-1, 2))
        augmented_audio_segment = AudioSegment(
            data=augmented_audio,
            sample_width=2,
            frame_rate=sample_rate,
            channels=channels
        )
        augmented_audio_segments.append(augmented_audio_segment)

    return augmented_audio_segments


def augment_pitch(audio_segments: List[AudioSegment]) -> List[AudioSegment]:
    transform = Compose(
        transforms=[
            PitchShift(p=1),
        ],
        p=1,
    )
    return augment(audio_segments, transform)


def augment_tempo(audio_segments: List[AudioSegment]) -> List[AudioSegment]:
    transform = Compose(
        transforms=[
            TimeStretch(p=1, leave_length_unchanged=False),
        ],
        p=1,
    )
    return augment(audio_segments, transform)


def augment_all(source_directory: str, output_directory: str):
    files = get_full_track_files(source_directory)
    
    for filename in tqdm(files, "Augmenting audio tracks"):
        stem_file_path = os.path.join(source_directory, 'stem.ogg')
        audio = AudioSegment.from_ogg(filename)
        stem = AudioSegment.from_ogg(stem_file_path)

        file_dir = os.path.dirname(filename)
        relative_path = os.path.relpath(file_dir, source_directory)
        output_file_path = os.path.join(output_directory, relative_path + '-orig')
        full_track_output_file_path = os.path.join(output_file_path, os.path.basename(filename))
        stem_output_file_path = os.path.join(output_file_path, os.path.basename(stem_file_path))
        if not os.path.exists(full_track_output_file_path) or not os.path.exists(stem_output_file_path):
            audio.export(full_track_output_file_path)
            stem.export(stem_output_file_path)

        for i in range(PITCH_AUGMENTATIONS_COUNT):
            file_dir = os.path.dirname(filename)
            relative_path = os.path.relpath(file_dir, source_directory)
            output_file_path = os.path.join(output_directory, relative_path + f'-pitch{i}')
            os.makedirs(output_file_path, exist_ok=True)
            full_track_output_file_path = os.path.join(output_file_path, os.path.basename(filename))
            stem_output_file_path = os.path.join(output_file_path, os.path.basename(stem_file_path))
            if not os.path.exists(full_track_output_file_path) or not os.path.exists(stem_output_file_path):
                augmented = augment_pitch([audio, stem])
                augmented[0].export(full_track_output_file_path)
                augmented[1].export(stem_output_file_path)

        for i in range(TEMPO_AUGMENTATIONS_COUNT):
            file_dir = os.path.dirname(filename)
            relative_path = os.path.relpath(file_dir, source_directory)
            output_file_path = os.path.join(output_directory, relative_path + f'-tempo{i}')
            os.makedirs(output_file_path, exist_ok=True)
            full_track_output_file_path = os.path.join(output_file_path, os.path.basename(filename))
            stem_output_file_path = os.path.join(output_file_path, os.path.basename(stem_file_path))
            if not os.path.exists(full_track_output_file_path) or not os.path.exists(stem_output_file_path):
                augmented = augment_tempo([audio, stem])
                augmented[0].export(full_track_output_file_path)
                augmented[1].export(stem_output_file_path)

    stem_files = get_stem_files(source_directory)
    for file_path in stem_files:
        relative_path = os.path.relpath(file_path, source_directory)
        full_track_output_file_path = os.path.join(output_directory, relative_path)
        os.makedirs(os.path.dirname(full_track_output_file_path), exist_ok=True)
        shutil.copy(file_path, full_track_output_file_path)
    
    return output_directory


if __name__ == '__main__':
    # random.seed(get_random_seed())
    augment_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR)