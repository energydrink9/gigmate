import glob
import os
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from audiomentations import Compose, AddGaussianSNR, BitCrush, BandStopFilter, RoomSimulator, SevenBandParametricEQ, TimeMask

from gigmate.utils.audio_utils import convert_audio_to_float_32, convert_audio_to_int_16

SOURCE_FILES_DIR = '/Users/michele/Music/soundstripe/merged'
OUTPUT_FILES_DIR = '/Users/michele/Music/soundstripe/augmented'
SKIP_AUGMENTATION_PROBABILITY = 0.2

def get_ogg_files(dir: str):
    return glob.glob(os.path.join(dir, '**/all-1.ogg'), recursive=True)

def augment(original_audio: AudioSegment) -> AudioSegment:
    sample_rate = original_audio.frame_rate
    channels = original_audio.channels
    audio = convert_audio_to_float_32(np.array(original_audio.get_array_of_samples()))
    transform = Compose(
        transforms=[
            AddGaussianSNR(min_snr_db=10., max_snr_db=50., p=0.15),
            #ApplyImpulseResponse(),
            BitCrush(min_bit_depth=5, max_bit_depth=10, p=0.2),
            #BandPassFilter(min_center_freq=200., max_center_freq=4000., p=1.0),
            BandStopFilter(min_center_freq=200., max_center_freq=4000., p=0.2),
            RoomSimulator(p=0.6, leave_length_unchanged=True),
            SevenBandParametricEQ(p=0.5, min_gain_db=-3.0, max_gain_db=3.0),
        ],
        p=0.8,
        shuffle=True
    )
    augmented_audio = transform(audio, sample_rate=sample_rate)
    data = convert_audio_to_int_16(augmented_audio)
    data = data.reshape((-1, 2))
    return AudioSegment(data=data, sample_width=2, frame_rate=sample_rate, channels=channels)


def augment_all(directory: str):
    files = get_ogg_files(directory)
    
    for filename in tqdm(files, "Augmenting audio tracks"):
        audio = AudioSegment.from_ogg(filename)
        augmented = augment(audio)
        augmented.export(filename)

if __name__ == '__main__':
    #random.seed(get_random_seed())
    augment_all(SOURCE_FILES_DIR)