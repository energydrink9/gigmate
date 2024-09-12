import random
import numpy as np
from symusic import Score, Note
from pretty_midi import PrettyMIDI
import os
from pydub import AudioSegment

def generate_random_dirname(prefix: str = 'tmp_', dir: str = '/tmp') -> str:
    """
    Generates a random directory name with a given prefix.

    Args:
        prefix (str, optional): The prefix for the directory name. Defaults to 'temp_'.

    Returns:
        str: A random directory name.
    """
    return f'{os.join(dir, prefix)}{random.randint(0, 1000000)}'

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

def calculate_mp3_length_in_seconds(file_path: str) -> float:
    """
    Calculates the length of an MP3 file in seconds.

    Args:
        file_path (str): The path to the MP3 file.

    Returns:
        float: The length of the MP3 file in seconds.
    """
    return len(AudioSegment.from_mp3(file_path)) / 1000

def calculate_wav_length_in_seconds(file_path: str) -> float:
    """
    Calculates the length of a WAV file in seconds.

    Args:
        file_path (str): The path to the WAV file.

    Returns:
        float: The length of the WAV file in seconds.
    """
    return len(AudioSegment.from_wav(file_path)) / 1000

def cut_mp3(file_path: str, end: int, out_file_path: str) -> None:
    """
    Cuts an MP3 file up to a specified end time and saves it to a new file.

    Args:
        file_path (str): The path to the original MP3 file.
        end (int): The end time in milliseconds.
        out_file_path (str): The path where the cut MP3 file will be saved.
    """
    cut = AudioSegment.from_mp3(file_path)[0:end]
    cut.export(out_file_path, format='mp3')
    return out_file_path

def calculate_score_length_in_seconds(score: Score) -> float:
    """
    Calculates the length of a musical score in seconds.

    Args:
        score (Score): The musical score.

    Returns:
        float: The length of the score in seconds.
    """
    score_end = score.end()
    score_tpq = score.tpq
    tempos = score.tempos

    total_seconds = 0
    current_tick = 0
    current_tempo_index = 0

    while current_tick < score_end and current_tempo_index < len(tempos):
        current_tempo = tempos[current_tempo_index]
        next_tempo_tick = score_end if current_tempo_index == len(tempos) - 1 else tempos[current_tempo_index + 1].time
        
        # Calculate ticks in this tempo section
        section_ticks = min(next_tempo_tick, score_end) - current_tick
        
        # Calculate duration of this section
        bpm = current_tempo.qpm
        seconds_per_beat = 60 / bpm
        quarter_notes = section_ticks / score_tpq
        section_seconds = quarter_notes * seconds_per_beat
        
        total_seconds += section_seconds
        
        current_tick = next_tempo_tick
        current_tempo_index += 1

    return total_seconds

def convert_pretty_midi_to_score(pretty_midi: PrettyMIDI) -> Score:
    """
    Converts a PrettyMIDI object to a Score object.

    Args:
        pretty_midi (PrettyMIDI): The PrettyMIDI object to convert.

    Returns:
        Score: The converted Score object.
    """
    temp_file = generate_random_filename(extension='.mid')
    try:
        pretty_midi.write(temp_file)
        return Score(temp_file)
    finally:
        os.remove(temp_file)

def convert_score_to_pretty_midi(score: Score) -> PrettyMIDI:
    """
    Converts a Score object to a PrettyMIDI object.

    Args:
        score (Score): The Score object to convert.

    Returns:
        PrettyMIDI: The converted PrettyMIDI object.
    """
    temp_file = generate_random_filename(extension='.mid')
    try:
        score.dump_midi(temp_file)
        return PrettyMIDI(temp_file)
    finally:
        os.remove(temp_file)

def pad_score_to_length(score: Score, target_length: float) -> Score:
    """
    Pads a musical score to a target length by adding silence.

    Args:
        score (Score): The musical score to pad.
        target_length (float): The target length in seconds.

    Returns:
        PrettyMIDI: The padded musical score as a PrettyMIDI object.
    """
    current_length = calculate_score_length_in_seconds(score)
    current_end = score.end()
    target_end = target_length * current_end / current_length
    
    if current_end < target_end:
        track = score.tracks[0] # Assuming we pad with the first track
        silence = Note(velocity=0, pitch=60, time=track.end(), duration=int(target_end - track.end()))
        track.notes.append(silence)

    return score

def convert_audio_to_int_16(audio_data: np.ndarray) -> np.ndarray:
    max_16bit = 2**15
    raw_data = audio_data * max_16bit
    raw_data = raw_data.astype(np.int16)
    return raw_data