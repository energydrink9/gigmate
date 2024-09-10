import numpy as np
from gigmate.audio_utils import generate_random_filename
from gigmate.constants import get_params
from gigmate.midi_conversion import convert_wav_to_midi
from gigmate.model import get_model
from gigmate.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.play import convert_to_int_16, get_audio_to_play
from gigmate.prediction import complete_midi
from gigmate.tokenizer import get_tokenizer
from gigmate.device import get_device
from scipy.io import wavfile

OUTPUT_TOKENS = 100
SAMPLE_RATE = 22050
device = get_device()
model = get_model(checkpoint_path=get_latest_model_checkpoint_path(), device=device)
tokenizer = get_tokenizer()
max_seq_len = get_params()['max_seq_len']
TEST_TIMING_FILE = 'resources/test_timing.wav'

def test_timing_of_generated_audio():

    # take a audio file
    _, original_audio = wavfile.read(TEST_TIMING_FILE)

    midi_from_wav = convert_wav_to_midi(TEST_TIMING_FILE)
    midi_from_wav_file = generate_random_filename(extension='.mid')
    midi_from_wav.write(midi_from_wav_file)
    midi_from_wav_length = midi_from_wav.get_end_time()

    # predict the next tokens
    prediction_with_input_file = complete_midi(model, device, midi_from_wav_file, tokenizer, max_seq_len, verbose=False, include_input=False, max_output_tokens=OUTPUT_TOKENS, temperature=0)

    # get the audio to play
    generated_audio = get_audio_to_play(prediction_with_input_file, 0, midi_from_wav_length, sample_rate=SAMPLE_RATE, get_current_time=lambda: 0)
    if generated_audio is not None:
        generated_audio = convert_to_int_16(generated_audio)

    # join original and generated audio
    merged_audio = np.concatenate((original_audio, generated_audio))
    
    # compute the total duration of the merged audio
    total_duration = len(merged_audio) / SAMPLE_RATE

    print(f"Total duration of the merged audio: {total_duration:.2f} seconds")
    print(f"Original audio duration: {len(original_audio) / SAMPLE_RATE:.2f} seconds")
    print(f"Generated audio duration: {len(generated_audio) / SAMPLE_RATE:.2f} seconds")

    # generate a random filename for the output
    output_file = generate_random_filename(extension='.wav')

    # export the merged audio to a wav file
    wavfile.write(output_file, SAMPLE_RATE, merged_audio)

    print(f'Merged audio file name: {output_file}')

    # TODO: check if the beat is in sync
    assert(False == True)


def test_timing_of_generated_audio_with_processing_time():

    # take a audio file
    _, complete_audio = wavfile.read(TEST_TIMING_FILE)

    # remove 1.5 seconds from the original audio
    original_audio = complete_audio[:-int(SAMPLE_RATE * 1.5)]

    midi_from_wav = convert_wav_to_midi(TEST_TIMING_FILE)
    midi_from_wav_file = generate_random_filename(extension='.mid')
    midi_from_wav.write(midi_from_wav_file)
    midi_from_wav_length = midi_from_wav.get_end_time()

    # predict the next tokens
    prediction_with_input_file = complete_midi(model, device, midi_from_wav_file, tokenizer, max_seq_len, verbose=False, include_input=True, max_output_tokens=OUTPUT_TOKENS, temperature=0)

    # get the audio to play
    generated_audio = get_audio_to_play(prediction_with_input_file, 0, midi_from_wav_length, sample_rate=SAMPLE_RATE, get_current_time=lambda: 1.5)
    if generated_audio is not None:
        generated_audio = convert_to_int_16(generated_audio)

    # join original and generated audio
    merged_audio = np.concatenate((complete_audio, generated_audio))
    
    # compute the total duration of the merged audio
    total_duration = len(merged_audio) / SAMPLE_RATE

    print(f"Total duration of the merged audio: {total_duration:.2f} seconds")
    print(f"Original audio duration: {len(original_audio) / SAMPLE_RATE:.2f} seconds")
    print(f"Generated audio duration: {len(generated_audio) / SAMPLE_RATE:.2f} seconds")

    # generate a random filename for the output
    output_file = generate_random_filename(extension='.wav')

    # export the merged audio to a wav file
    wavfile.write(output_file, SAMPLE_RATE, merged_audio)

    print(f'Merged audio file name: {output_file}')

    # TODO: check if the beat is in sync
    assert(False == True)