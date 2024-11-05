import glob
import multiprocessing
import os
import shutil
from typing import List, Tuple
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from audiomentations import Compose, AddGaussianSNR, BitCrush, BandStopFilter, RoomSimulator, SevenBandParametricEQ

from gigmate.utils.audio_utils import clamp_audio_data, convert_audio_to_float_32, convert_audio_to_int_16

SOURCE_FILES_DIR = '../dataset/augmented'
OUTPUT_FILES_DIR = '../dataset/distorted'


def get_full_track_files(dir: str):
    return glob.glob(os.path.join(dir, '**/all.ogg'), recursive=True)


def get_stem_file(dir: str):
    return os.path.join(dir, 'stem.ogg')


def get_files_pairs(dir: str) -> List[Tuple[str, str]]:
    full_track_files = get_full_track_files(dir)
    pairs = [(full_track_file, get_stem_file(os.path.dirname(full_track_file))) for full_track_file in full_track_files]
    return pairs


def distort_audio(original_audio: AudioSegment) -> AudioSegment:
    sample_rate = original_audio.frame_rate
    channels = original_audio.channels
    audio = convert_audio_to_float_32(np.array(original_audio.get_array_of_samples()))
    transform = Compose(
        transforms=[
            AddGaussianSNR(min_snr_db=10., max_snr_db=50., p=0.15),
            # ApplyImpulseResponse(),
            BitCrush(min_bit_depth=3, max_bit_depth=7, p=0.2),
            # BandPassFilter(min_center_freq=200., max_center_freq=4000., p=1.0),
            BandStopFilter(min_center_freq=200., max_center_freq=4000., p=0.2),
            RoomSimulator(p=0.6, leave_length_unchanged=True),
            SevenBandParametricEQ(p=0.5, min_gain_db=-3.5, max_gain_db=3.5),
        ],
        p=0.8,
        shuffle=True
    )
    augmented_audio = transform(audio, sample_rate=sample_rate)
    data = convert_audio_to_int_16(clamp_audio_data(augmented_audio))
    data = data.reshape((-1, 2))
    return AudioSegment(data=data, sample_width=2, frame_rate=sample_rate, channels=channels)  # type: ignore


def distort(params: Tuple[Tuple[str, str], str, str]) -> None:
    (full_track_file_path, stem_file_path), source_directory, output_directory = params
    file_dir = os.path.dirname(full_track_file_path)
    full_track_relative_path = os.path.relpath(file_dir, source_directory)
    actual_output_dir = os.path.join(output_directory, full_track_relative_path)
    os.makedirs(actual_output_dir, exist_ok=True)
    full_track_output_file_path = os.path.join(actual_output_dir, os.path.basename(full_track_file_path))

    if not os.path.exists(full_track_output_file_path):
        audio = AudioSegment.from_ogg(full_track_file_path)
        augmented = distort_audio(audio)
        augmented.export(full_track_output_file_path)

    stem_relative_path = os.path.relpath(stem_file_path, source_directory)
    stem_output_file_path = os.path.join(output_directory, stem_relative_path)
    if not os.path.exists(stem_output_file_path):
        shutil.copy(stem_file_path, stem_output_file_path)


def distort_all(source_directory: str, output_directory: str):
    files: List[Tuple[str, str]] = get_files_pairs(source_directory)
    
    params_list: List[Tuple[Tuple[str, str], str, str]] = [(file_pair, source_directory, output_directory) for file_pair in files]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(distort, params_list), total=len(params_list), desc="Distorting audio tracks"))
    
    return output_directory


if __name__ == '__main__':
    distort_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR)