import random
import numpy as np
import os
from pydub import AudioSegment
import torch

SOUNDFONT_PATH = 'output/Roland SOUNDCanvas SC-55 Up.sf2'  # Downloaded from https://archive.org/download/free-soundfonts-sf2-2019-04


def generate_random_dirname(prefix: str = 'tmp_', dir: str = '/tmp') -> str:
    """
    Generates a random directory name with a given prefix.

    Args:
        prefix (str, optional): The prefix for the directory name. Defaults to 'temp_'.

    Returns:
        str: A random directory name.
    """
    return f'{os.path.join(dir, prefix)}{random.randint(0, 1000000)}'


def generate_random_filename(prefix: str = 'tmp_', extension: str = '.wav', dir: str = '/tmp') -> str:
    """
    Generates a random filename with a given prefix, extension, and directory.

    Args:
        prefix (str, optional): The prefix for the filename. Defaults to 'temp_'.
        extension (str, optional): The file extension. Defaults to '.wav'.
        dir (str, optional): The directory where the file will be saved. Defaults to 'output'.

    Returns:
        str: A random filename.
    """
    return f'{os.path.join(dir, prefix)}{random.randint(0, 1000000)}{extension}'


def calculate_audio_length_in_seconds(file_path: str) -> float:
    """
    Calculates the length of an audio file in seconds.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        float: The length of the audio file in seconds.
    """
    return len(AudioSegment.from_file(file_path)) / 1000


def cut_audio(audio: AudioSegment, end: int, out_file_path: str) -> str:
    """
    Cuts an audio wav file up to a specified end time and saves it to a new file.

    Args:
        file_path (str): The path to the original wav file.
        end (int): The end time in milliseconds.
        out_file_path (str): The path where the cut file will be saved.
    """
    if len(audio) > end:
        audio = audio[0:end]

    audio.export(out_file_path, format='wav')
    return out_file_path


def calculate_audio_length(audio: np.ndarray, sample_rate: int) -> float:
    duration = len(audio) / sample_rate
    return duration


def calculate_audio_tensor_length(audio: torch.Tensor, frame_rate: int) -> float:
    return audio.shape[-1] / frame_rate


def clamp_audio_data(audio_data: np.ndarray) -> np.ndarray:
    return np.clip(audio_data, -1, 1)


def convert_audio_to_int_16(audio_data: np.ndarray) -> np.ndarray:
    max_16bit = 2**15 - 1
    assert audio_data.max() <= 1 and audio_data.min() >= -1, f'Overflow error during audio conversion: {audio_data.max()} vs {1}'
    raw_data = audio_data * max_16bit
    assert raw_data.max() <= max_16bit, f'Overflow error during audio conversion: {raw_data.max()} vs {max_16bit}'
    return raw_data.astype(np.int16)


def convert_audio_to_float_32(audio_data: np.ndarray) -> np.ndarray:
    max_32bit = 2**31 - 1
    assert audio_data.max() <= max_32bit, f'Overflow error during audio conversion: {audio_data.max()} vs {max_32bit}'
    raw_data = audio_data / max_32bit
    return raw_data.astype(np.float32)
