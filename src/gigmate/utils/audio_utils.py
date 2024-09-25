import random
from typing import Optional
import numpy as np
from symusic.types import Score
from symusic import Score as ScoreFactory, Note as NoteFactory
from pretty_midi import PrettyMIDI
import os
from pydub import AudioSegment

SOUNDFONT_PATH = 'output/Roland SOUNDCanvas SC-55 Up.sf2'# Downloaded from https://archive.org/download/free-soundfonts-sf2-2019-04

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

def cut_audio(audio: AudioSegment, end: int, out_file_path: str) -> None:
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

def calculate_audio_length(audio: np.ndarray, sample_rate: int) -> float:
    duration = len(audio) / sample_rate
    return duration

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
        return ScoreFactory(temp_file)
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
        silence = NoteFactory(velocity=0, pitch=60, time=track.end(), duration=int(target_end - track.end()))
        track.notes.append(silence)

    return score

def cut_score_to_length(score: Score, target_length: float) -> Score:
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


    if current_end <= target_end:
        return score

    for track in score.tracks:
        for note in track.notes:
            if note.time + note.duration > target_end:
                note.duration = target_end - note.time
                if note.duration < 0:
                    track.notes.remove(note)

    return score

def convert_audio_to_int_16(audio_data: np.ndarray) -> np.ndarray:
    max_16bit = 2**15 - 1
    raw_data = audio_data * max_16bit
    return raw_data.astype(np.int16)

def convert_audio_to_float_32(audio_data: np.ndarray) -> np.ndarray:
    max_32bit = 2**31 - 1
    raw_data = audio_data / max_32bit
    return raw_data.astype(np.float32)

def get_program_midi(predicted_midi: PrettyMIDI, midi_program: int) -> PrettyMIDI:
    filtered_midi = PrettyMIDI()

    instrument_code = midi_program if midi_program != -1 else 0
    
    # Iterate through all instruments in the predicted MIDI
    for instrument in predicted_midi.instruments:
        if instrument.program == instrument_code:
            # Add the instrument to the filtered MIDI
            filtered_midi.instruments.append(instrument)

    return filtered_midi

def synthesize_midi(midi: PrettyMIDI, midi_program: Optional[int], sample_rate: int, soundfont_path=SOUNDFONT_PATH) -> np.ndarray:
    program_midi = get_program_midi(midi, midi_program) if midi_program is not None else midi
    sf2_path = SOUNDFONT_PATH if os.path.exists(soundfont_path) else None
    return program_midi.fluidsynth(fs=sample_rate, sf2_path=sf2_path)