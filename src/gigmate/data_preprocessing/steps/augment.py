import glob
import os
import shutil
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from audiomentations import Compose, AddGaussianSNR, BitCrush, BandStopFilter, RoomSimulator, SevenBandParametricEQ

from gigmate.utils.audio_utils import clamp_audio_data, convert_audio_to_float_32, convert_audio_to_int_16

SOURCE_FILES_DIR = '/Users/michele/Music/soundstripe/merged'
OUTPUT_FILES_DIR = '/Users/michele/Music/soundstripe/augmented'


def get_full_track_files(dir: str):
    return glob.glob(os.path.join(dir, '**/all-*.ogg'), recursive=True)


def get_stem_files(dir: str):
    return glob.glob(os.path.join(dir, '**/stem.ogg'), recursive=True)


def augment(original_audio: AudioSegment) -> AudioSegment:
    sample_rate = original_audio.frame_rate
    channels = original_audio.channels
    audio = convert_audio_to_float_32(np.array(original_audio.get_array_of_samples()))
    transform = Compose(
        transforms=[
            AddGaussianSNR(min_snr_db=10., max_snr_db=50., p=0.15),
            # ApplyImpulseResponse(),
            BitCrush(min_bit_depth=5, max_bit_depth=10, p=0.2),
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
    return AudioSegment(data=data, sample_width=2, frame_rate=sample_rate, channels=channels)


def augment_all(source_directory: str, output_directory: str):
    files = get_full_track_files(source_directory)
    
    for filename in tqdm(files, "Augmenting audio tracks"):
        audio = AudioSegment.from_ogg(filename)
        augmented = augment(audio)
        file_dir = os.path.dirname(filename)
        relative_path = os.path.relpath(file_dir, source_directory)
        output_file_path = os.path.join(output_directory, relative_path)
        os.makedirs(output_file_path, exist_ok=True)
        output_file_path = os.path.join(output_file_path, os.path.basename(filename))
        augmented.export(output_file_path)

    stem_files = get_stem_files(source_directory)
    for file_path in stem_files:
        relative_path = os.path.relpath(file_path, source_directory)
        output_file_path = os.path.join(output_directory, relative_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        shutil.copy(file_path, output_file_path)
    
    return output_directory


if __name__ == '__main__':
    # random.seed(get_random_seed())
    augment_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR)