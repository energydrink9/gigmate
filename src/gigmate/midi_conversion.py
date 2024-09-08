import os
from basic_pitch import build_icassp_2022_model_path, FilenameSuffix
from basic_pitch.inference import predict as basic_pitch_predict, Model
from scipy.io import wavfile
from pretty_midi import PrettyMIDI
from gigmate.audio_utils import generate_random_filename

OUTPUT_SAMPLE_RATE = 22050

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

def convert_wav_to_midi(audio_file: str) -> PrettyMIDI:
    """
    Converts an audio file to a MIDI file using the Basic Pitch model.

    This function takes an audio file, converts it to MIDI using the Basic Pitch model,
    and returns the MIDI object.

    Args:
        audio_file (str): The path to the audio file to be converted.

    Returns:
        pretty_midi.PrettyMIDI: The MIDI object created.
    """
    global basic_pitch_model

    _, midi_data, _ = basic_pitch_predict(audio_file, basic_pitch_model, onset_threshold=2.0, minimum_note_length=80)
    temp_file = generate_random_filename(extension='.mid')
    midi_data.write(temp_file)
    midi = PrettyMIDI(temp_file)
    os.remove(temp_file)
    
    return midi