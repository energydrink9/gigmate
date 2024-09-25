import multiprocessing
import os
import shutil
import tempfile
import numpy as np
from basic_pitch import build_icassp_2022_model_path, FilenameSuffix
from basic_pitch.inference import predict as basic_pitch_predict, Model
from scipy.io import wavfile
from pretty_midi import PrettyMIDI
import spleeter.separator
from symusic.types import Score, Track
from symusic import Score as ScoreFactory
from concurrent.futures import ThreadPoolExecutor
from gigmate.utils.audio_utils import calculate_audio_length_in_seconds, convert_audio_to_float_32, convert_audio_to_int_16, cut_audio, cut_score_to_length, generate_random_dirname, generate_random_filename, pad_score_to_length
from pydub import AudioSegment

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
    audio_length = calculate_audio_length_in_seconds(file_path)

    with tempfile.NamedTemporaryFile(suffix='.mid', delete=True) as temp_file:
        midi_data.write(temp_file.name)
        temp_file.flush()
        midi = ScoreFactory(temp_file.name)
        midi = cut_score_to_length(midi, audio_length)
        midi = pad_score_to_length(midi, audio_length)
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

def separate_audio_tracks(audio_file: AudioSegment, device: str, separator: spleeter.separator.Separator) -> list[tuple[str, AudioSegment]]:
    """
    Separates an audio file into its individual tracks using the Demucs model.

    This function takes an audio file as input, separates it into its individual tracks using the Demucs model,
    and a list of tuples containing the instrument name of each track and the audio.

    Args:
        audio_file (AudioSegment): The audio file to be separated.

    Returns:
        tuple[str, list[tuple[str, AudioSegment]]]: A list of tuples containing the stems (instrument, track)
    """
    # The separator requires a tensor composed of 2 channels
    two_channels_audio_file = AudioSegment.from_mono_audiosegments(audio_file, audio_file) if audio_file.channels == 1 else audio_file
    tensor = convert_audio_to_float_32(np.array(two_channels_audio_file.get_array_of_samples())).reshape((-1, 2)) # reshape 2, -1 for demuc
    #tensor = torch.tensor(convert_audio_to_float_32(np.array(two_channels_audio_file.get_array_of_samples()))).reshape((2, -1))
    #_, stems = separator.separate_tensor(tensor, audio_file.frame_rate)
    stems = separator.separate(tensor)

    def get_audio_segment_demucs(track):
        data = convert_audio_to_int_16(track.detach().cpu().numpy().reshape(-1))
        data = convert_audio_to_int_16(track.reshape(-1))
        return AudioSegment(data=data, sample_width=2, frame_rate=separator.model.samplerate, channels=separator.model.audio_channels)

    def get_audio_segment_spleeter(track):
        data = convert_audio_to_int_16(track.reshape(-1))
        return AudioSegment(data=data, sample_width=2, frame_rate=audio_file.frame_rate, channels=audio_file.channels)

    return [(instrument_name, get_audio_segment_spleeter(track)) for instrument_name, track in stems.items()]

def merge_midis(midis: list[tuple[str, Score]]) -> Score:
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
    def update_track(instrument_name: str, track: Track):
        track.program = get_program_code(instrument_name)
        track.is_drum = instrument_name == 'drums'
        track.name = instrument_name.capitalize()
        return track
    first_score = midis[0][1]
    score = ScoreFactory.from_tpq(first_score.tpq)
    score.tempos = first_score.tempos
    tracks = [update_track(instrument_name, track) for (instrument_name, midi_score) in midis for track in list(midi_score.tracks)]
    score.tracks = tracks
    return score

def _convert_stem(instrument_name: str, file_path:str):
    return instrument_name, convert_wav_to_midi(file_path)

def convert_stems_to_midi(stems: dict[str, str], executor) -> dict[str, Score]:
    return dict(executor.map(lambda item: _convert_stem(item[0], item[1]), stems.items()))

def convert_audio_to_midi(audio_file: AudioSegment, device: str, separator) -> Score:
    """
    Converts an audio file to MIDI format.

    This function takes an audio file path as input, separates the audio into its individual tracks,
    converts each track to MIDI format, and then merges the MIDI tracks into a single MIDI file.

    Args:
        file_path (str): The path to the audio file to be converted.

    Returns:
        symusic.Score: The merged MIDI file.
    """
    temp_dir = generate_random_dirname()
    os.makedirs(temp_dir, exist_ok=True)
    original_audio_length = len(audio_file) / 1000
    separate_tracks = separate_audio_tracks(audio_file, device, separator)
    # Cut tracks to original audio length to ensure length consistency
    executor = ThreadPoolExecutor(min(len(cut_tracks), multiprocessing.cpu_count()))
    cut_tracks = dict({instrument_name: cut_audio(track, int(original_audio_length * 1000), os.path.join(temp_dir, instrument_name + '.wav')) for instrument_name, track in separate_tracks})
    try:
        midi_tracks = convert_stems_to_midi(cut_tracks, executor)
        merged_score = merge_midis(list(midi_tracks.items()))
        score = pad_score_to_length(merged_score, original_audio_length) # Pad score to ensure length consistency with original audio
        return score
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)