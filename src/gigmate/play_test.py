from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import tempfile
from typing import cast
from miditok import TokSequence
import numpy as np
import pytest
from gigmate.utils.audio_utils import calculate_audio_length, calculate_score_length_in_seconds, convert_audio_to_float_32, generate_random_filename, synthesize_midi
from gigmate.utils.constants import get_params
from gigmate.domain.midi_conversion import convert_stems_to_midi, convert_wav_to_midi, merge_midis
from gigmate.model.model import get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.play import get_audio_to_play
from gigmate.domain.prediction import complete_sequence
from gigmate.model.tokenizer import get_tokenizer
from gigmate.utils.device import get_device
from scipy.io import wavfile
from spleeter.separator import Separator
import soundfile as sf
from symusic.types import Score
from pretty_midi import PrettyMIDI

SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 44100
device = get_device()
model = get_model(checkpoint_path=get_latest_model_checkpoint_path(), device=device)
tokenizer = get_tokenizer()
max_seq_len = get_params()['max_seq_len']
TEST_GENERATION_FILE = 'resources/test_generation.wav'
TEST_TIMING_FILE = 'resources/test_timing_2.wav'
MIDI_PROGRAM = None

def get_temporary_file(audio: np.ndarray, output_sample_rate: int) -> tempfile._TemporaryFileWrapper:
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.wav', delete=False) as file:
        sf.write(file, audio, output_sample_rate, format='WAV')
        file.flush()  # Ensure the data is written to the file
        file.seek(0)  # Reset the file pointer to the beginning
        return file

def test_step_separation_returns_tracks_of_equal_length():
    sample_rate, original_audio = wavfile.read(TEST_GENERATION_FILE)
    original_audio = convert_audio_to_float_32(original_audio)
    original_audio_duration = calculate_audio_length(original_audio, sample_rate)
    print(f'Original audio duration: {original_audio_duration}')

    separator = Separator('spleeter:5stems')
    stems = separator.separate(original_audio)
    
    for stem_name, stem in stems.items():
        audio_duration = calculate_audio_length(stem, sample_rate)
        print(f'Stem duration: {audio_duration}')
        assert audio_duration == original_audio_duration, f'Stem {stem_name} duration {audio_duration} is not equal to original audio duration {original_audio_duration}'

def test_midi_generation_returns_midi_of_equal_length():
    sample_rate, original_audio = wavfile.read(TEST_GENERATION_FILE)
    original_audio = convert_audio_to_float_32(original_audio)
    original_audio_duration = calculate_audio_length(original_audio, sample_rate)

    score = convert_wav_to_midi(TEST_GENERATION_FILE)
    midi_audio_duration = calculate_score_length_in_seconds(score)

    assert pytest.approx(midi_audio_duration, rel=1e-3) == original_audio_duration, f'Midi duration {midi_audio_duration} is not equal to original audio duration {original_audio_duration}'

def test_generation_of_audio():

    # take an audio file
    sample_rate, original_audio = wavfile.read(TEST_GENERATION_FILE)
    original_audio = convert_audio_to_float_32(original_audio)
    original_audio_duration = calculate_audio_length(original_audio, sample_rate)
    print(f'Original audio duration: {original_audio_duration}')

    separator = Separator('spleeter:5stems')
    stems = separator.separate(original_audio)
    
    executor = ThreadPoolExecutor(multiprocessing.cpu_count())
    file_stems = dict({stem[0]: get_temporary_file(stem[1], sample_rate).name for stem in stems.items()})
    midis = convert_stems_to_midi(file_stems, executor)
    score = merge_midis(list(midis.items()))
    temp_file_name = generate_random_filename(extension='.mid')
    score.dump_midi(temp_file_name)
    midi = PrettyMIDI(temp_file_name)
    os.remove(temp_file_name)

    token_sequence: TokSequence = cast(TokSequence, tokenizer.encode(score))
    input_sequence = cast(list[int], token_sequence.ids)

    output_sequence = complete_sequence(model, device, tokenizer, input_sequence, max_seq_len=max_seq_len, max_output_tokens=200, max_output_length_in_seconds=5, show_progress=False)
    sequence: Score = tokenizer.decode(output_sequence)
    temp_file_name = generate_random_filename(extension='.mid')
    sequence.dump_midi(temp_file_name)
    midi = PrettyMIDI(temp_file_name)
    os.remove(temp_file_name)
    data = synthesize_midi(midi, MIDI_PROGRAM, OUTPUT_SAMPLE_RATE)
    wavfile.write(temp_file_name.replace('.mid', '.wav'), sample_rate, data)

    print(f"Saved file {temp_file_name.replace('.mid', '.wav')}")
 
def test_timing_of_generated_audio():

    # take a audio file
    original_audio_sample_rate, original_audio = wavfile.read(TEST_TIMING_FILE)
    original_audio = convert_audio_to_float_32(original_audio)

    separator = Separator('spleeter:5stems')
    stems = separator.separate(original_audio)
    executor = ThreadPoolExecutor(multiprocessing.cpu_count())
    file_stems = dict({stem[0]: get_temporary_file(stem[1], original_audio_sample_rate).name for stem in stems.items()})
    midis = convert_stems_to_midi(file_stems, executor)
    score = merge_midis(list(midis.items()))
    token_sequence: TokSequence = cast(TokSequence, tokenizer.encode(score))
    input_sequence = cast(list[int], token_sequence.ids)
    output_sequence = complete_sequence(model, device, tokenizer, input_sequence, max_seq_len=max_seq_len, max_output_tokens=200, max_output_length_in_seconds=8, show_progress=False)
    sequence: Score = tokenizer.decode(output_sequence)
    temp_file_name = generate_random_filename(extension='.mid')
    sequence.dump_midi(temp_file_name)
    midi = PrettyMIDI(temp_file_name)
    data = synthesize_midi(midi, MIDI_PROGRAM, OUTPUT_SAMPLE_RATE)
    wavfile.write(temp_file_name.replace('.mid', '.wav'), original_audio_sample_rate, data)

    print(f"Saved file {temp_file_name.replace('.mid', '.wav')}")

    # TODO: check if the beat is in sync
    assert False == True, 'Beat not in sync'


def test_timing_of_generated_audio_with_processing_time():
        
    # take a audio file
    original_audio_sample_rate, original_audio = wavfile.read(TEST_TIMING_FILE)
    original_audio = convert_audio_to_float_32(original_audio)

    separator = Separator('spleeter:5stems')
    stems = separator.separate(original_audio)
    executor = ThreadPoolExecutor(multiprocessing.cpu_count())
    file_stems = dict({stem[0]: get_temporary_file(stem[1], original_audio_sample_rate).name for stem in stems.items()})
    midis = convert_stems_to_midi(file_stems, executor)
    score = merge_midis(list(midis.items()))
    token_sequence: TokSequence = cast(TokSequence, tokenizer.encode(score))
    input_sequence = cast(list[int], token_sequence.ids)
    output_sequence = complete_sequence(model, device, tokenizer, input_sequence, max_seq_len=max_seq_len, max_output_tokens=100, max_output_length_in_seconds=10, show_progress=True)
    sequence: Score = tokenizer.decode(output_sequence)
    temp_file_name = generate_random_filename(extension='.mid')
    sequence.dump_midi(temp_file_name)
    midi = PrettyMIDI(temp_file_name)
    data = synthesize_midi(midi, MIDI_PROGRAM, OUTPUT_SAMPLE_RATE)
    wavfile.write(temp_file_name.replace('.mid', '.wav'), original_audio_sample_rate, data)

    print(f"Saved file {temp_file_name.replace('.mid', '.wav')}")
    # not let's cut the audio data removing (chunk.record_end_time - chunk.record_start_time) at the beginning:
    samples_to_remove = int((chunk.record_end_time - chunk.record_start_time) * OUTPUT_SAMPLE_RATE)
    # Remove the samples from the beginning of the audio data
    data = data[samples_to_remove:]
    
#     # take a audio file
#     _, complete_audio = wavfile.read(TEST_TIMING_FILE)

#     # remove 1.5 seconds from the original audio
#     original_audio = complete_audio[:-int(SAMPLE_RATE * 1.5)]

#     midi_from_wav = convert_wav_to_midi(TEST_TIMING_FILE)
#     midi_from_wav_file = generate_random_filename(extension='.mid')
#     midi_from_wav.dump_midi(midi_from_wav_file)
#     midi_from_wav_length = calculate_score_length_in_seconds(midi_from_wav)

#     # predict the next tokens
#     prediction_with_input_file = complete_midi(model, device, midi_from_wav_file, tokenizer, max_seq_len, verbose=False, include_input=True, max_output_tokens=OUTPUT_TOKENS, temperature=0)

#     # get the audio to play
#     generated_audio = get_audio_to_play(prediction_with_input_file, 0, midi_from_wav_length, sample_rate=SAMPLE_RATE, get_current_time=lambda: 1.5)
#     if generated_audio is not None:
#         generated_audio = convert_audio_to_int_16(generated_audio)

#     # join original and generated audio
#     merged_audio = np.concatenate((complete_audio, generated_audio))
    
#     # compute the total duration of the merged audio
#     total_duration = len(merged_audio) / SAMPLE_RATE

#     print(f"Total duration of the merged audio: {total_duration:.2f} seconds")
#     print(f"Original audio duration: {len(original_audio) / SAMPLE_RATE:.2f} seconds")
#     print(f"Generated audio duration: {len(generated_audio) / SAMPLE_RATE:.2f} seconds")

#     # generate a random filename for the output
#     output_file = generate_random_filename(extension='.wav')

#     # export the merged audio to a wav file
#     wavfile.write(output_file, SAMPLE_RATE, merged_audio)

#     print(f'Merged audio file name: {output_file}')

#     # TODO: check if the beat is in sync
#     assert(False == True)