import glob
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
import shlex
import shutil
import time
from basic_pitch import build_icassp_2022_model_path, FilenameSuffix
from basic_pitch.inference import predict as basic_pitch_predict, Model
import demucs.separate
from scipy.io import wavfile
from pretty_midi import PrettyMIDI
from symusic import Score
from gigmate.utils.audio_utils import calculate_mp3_length_in_seconds, cut_mp3, generate_random_dirname, generate_random_filename, pad_score_to_length

OUTPUT_SAMPLE_RATE = 22050
DEFAULT_PROGRAM_CODE = 0
GUITAR_PROGRAM_CODE = 25
BASS_PROGRAM_CODE = 33
VOCALS_PROGRAM_CODE = 86
OTHER_PROGRAM_CODE = 0
PIANO_PROGRAM_CODE = 0
DRUMS_PROGRAM_CODE = 0

basic_pitch_model = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))

def convert_midi_to_wav(midi: PrettyMIDI, fs: int = OUTPUT_SAMPLE_RATE) -> str:
    """
    Converts a MIDI object to a WAV file.

    This function takes a MIDI object and a sample rate, synthesizes the MIDI data
    into audio using fluidsynth, and saves the result as a temporary WAV file.

    Args:
        midi (pretty_midi.PrettyMIDI): The MIDI object to be converted.
        fs (int): The desired sample rate for the output WAV file.

    Returns:
        str: The path to the temporary WAV file created.

    Note:
        The function uses a global OUTPUT_SAMPLE_RATE for writing the WAV file,
        which may differ from the input fs parameter used for synthesis.
    """

    audio_data = midi.fluidsynth(fs=fs)
    temp_file = generate_random_filename(extension='.wav')
    wavfile.write(temp_file, OUTPUT_SAMPLE_RATE, audio_data)
    return temp_file

def convert_wav_to_midi(file_path: str) -> Score:
    """
    Converts an audio file to a MIDI file using the Basic Pitch model.

    This function takes an audio file, converts it to MIDI using the Basic Pitch model,
    and returns the MIDI object.

    Args:
        audio_file (str): The path to the audio file to be converted.

    Returns:
        symusic.Score: The MIDI object created.
    """
    global basic_pitch_model
    
    _, midi_data, _ = basic_pitch_predict(file_path, basic_pitch_model, onset_threshold=2.0, minimum_note_length=80)
    temp_file = generate_random_filename(extension='.mid')
    midi_data.write(temp_file)
    midi = Score(temp_file)
    os.remove(temp_file)
    
    return midi

def get_program_code(instrument_name: str) -> int:
    """
    Returns the MIDI program code for a given instrument name.

    Args:
        instrument_name (str): The name of the instrument.

    Returns:
        int: The MIDI program code for the instrument.
    """
    match instrument_name:
        case "guitar":
            return GUITAR_PROGRAM_CODE
        case "bass":
            return BASS_PROGRAM_CODE
        case "vocals":
            return VOCALS_PROGRAM_CODE
        case "other":
            return OTHER_PROGRAM_CODE
        case "piano":
            return PIANO_PROGRAM_CODE
        case "drums":
            return DRUMS_PROGRAM_CODE
        case _:
            return DEFAULT_PROGRAM_CODE

def separate_audio_tracks(filename: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Separates an audio file into its individual tracks using the Demucs model.

    This function takes an audio file as input, separates it into its individual tracks using the Demucs model,
    and returns the directory where the separated tracks are stored along with a list of tuples containing the
    instrument name of each track and its corresponding file path.

    Args:
        filename (str): The path to the audio file to be separated.

    Returns:
        tuple[str, list[tuple[str, str]]]: A tuple containing the directory path where the separated tracks are stored,
        and a list of tuples where each tuple contains the instrument name of a track and its file path.
    """
    temp_dir = generate_random_dirname()
    demucs.separate.main(shlex.split(f'--mp3 -n htdemucs_6s --out "{temp_dir}" "{filename}"'))
    return (temp_dir, [(os.path.splitext(os.path.basename(filename))[0], filename) for filename in glob.glob(os.path.join(temp_dir, '**', '*.mp3'), recursive=True)])

def merge_midi_files(files: list[tuple[str, Score]]) -> Score:
    """
    Merges multiple MIDI files into a single one.

    This function takes a directory name and a list of tuples, where each tuple contains an instrument name and a Score.
    It merges all the MIDI files into a single one, setting the program, drum status, and name of each instrument according to the tuple.

    Args:
        dirname (str): The directory name where the merged MIDI file will be stored.
        files (list[tuple[str, Score]]): A list of tuples, where each tuple contains an instrument name and a Score.

    Returns:
        symusic.Score: The merged MIDI file.
    """

    def update_track(instrument_name, track):
        track.program = get_program_code(instrument_name)
        track.is_drum = instrument_name == 'drums'
        track.name = instrument_name.capitalize()
        return track
    
    score = Score.from_tpq(files[0][1].tpq)
    score.tempos = files[0][1].tempos
    tracks = [update_track(instrument_name, track) for (instrument_name, score) in files for track in score.tracks]
    score.tracks = tracks

    return score

def _convert_entry(entry):
    instrument_name, file_path = entry
    return instrument_name, convert_wav_to_midi(file_path)

def convert_audio_to_midi(file_path: str) -> Score:
    """
    Converts an audio file to MIDI format.

    This function takes an audio file path as input, separates the audio into its individual tracks,
    converts each track to MIDI format, and then merges the MIDI tracks into a single MIDI file.

    Args:
        file_path (str): The path to the audio file to be converted.

    Returns:
        symusic.Score: The merged MIDI file.
    """
    original_audio_length = calculate_mp3_length_in_seconds(file_path)
    temp_dir, separate_tracks = separate_audio_tracks(file_path)
    # Cut tracks to original audio length to ensure length consistency
    cut_tracks = [(instrument_name, cut_mp3(track, int(original_audio_length * 1000), track)) for instrument_name, track in separate_tracks]

    try:
        midi_tracks = map(_convert_entry, cut_tracks)
        merged_score = merge_midi_files(midi_tracks)
        score = pad_score_to_length(merged_score, original_audio_length) # Pad score to ensure length consistency with original audio
        return score
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
