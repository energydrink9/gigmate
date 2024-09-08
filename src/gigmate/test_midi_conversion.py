# In the main function, record audio from the microphone, then, every N milliseconds, convert the audio to a MIDI file using basic pitch.
# Pass the MIDI file to the model, and get a prediction to complete the MIDI file. Play the completed MIDI file.

import io
import time

import requests
from gigmate.model import get_model
import sounddevice as sd
import torch
from basic_pitch.inference import predict as basic_pitch_predict, Model
from basic_pitch import build_icassp_2022_model_path, FilenameSuffix
from gigmate.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.predict import compute_output_sequence, get_tokenizer, get_params, get_device
from pretty_midi import PrettyMIDI
from multiprocessing import Process, Queue
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment

CHANNELS = 1
SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 22050
OUTPUT_TOKENS_COUNT = 100
BUFFER_SIZE_IN_SECONDS = 10
DEBUG = True

tokenizer = get_tokenizer()
max_seq_len = get_params()['max_seq_len']
basic_pitch_model = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))

def get_audio_buffer_length_in_seconds(audio_buffer):
    return len(audio_buffer) / SAMPLE_RATE

def listen(conversion_queue):
    incoming_audio = None
    
    def callback(indata, frames, audio_time, status):
        nonlocal incoming_audio

        sample_time = time.time() - (audio_time.currentTime - audio_time.inputBufferAdcTime)
        if incoming_audio is None:
            incoming_audio = {
                'audio_buffer': [indata.copy()],
                'audio_start_time': [sample_time]
            }
        else:
            incoming_audio['audio_buffer'].append(indata.copy())
            incoming_audio['audio_start_time'].append(sample_time)
        
        audio_buffer_oldness = time.time() - incoming_audio['audio_start_time'][0]

        if DEBUG:
            print(f"Audio buffer oldness: {audio_buffer_oldness}, len {len(incoming_audio['audio_buffer'])}")
        
        while audio_buffer_oldness > BUFFER_SIZE_IN_SECONDS:
            incoming_audio['audio_buffer'] = incoming_audio['audio_buffer'][1:]
            incoming_audio['audio_start_time'] = incoming_audio['audio_start_time'][1:]
            audio_buffer_oldness = time.time() - incoming_audio['audio_start_time'][0]

        audio_buffer = np.concatenate(incoming_audio['audio_buffer'], axis=0).reshape(-1, 1).astype(np.float32)
        audio_start_time = incoming_audio['audio_start_time'][0]

        try:
            conversion_queue.put((audio_buffer, audio_start_time), block=False)
        except Exception as e:
            print('Error while inserting audio buffer in queue')
            print(e)

    while True:
        # Decrease the block size to reduce latency
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE / 10), channels=CHANNELS, latency='low', callback=callback):
                sd.sleep(10000)
        except Exception as e:
            print(f"Error while listening: {e}")
            

def convert_audio_to_midi(conversion_queue, prediction_queue):

    start_time = time.time()
    # Write audio buffer to file
    audio_file = 'output/test_creep.mp3'

    
    audio = AudioSegment.from_mp3(audio_file)
    audio_length_seconds = len(audio) / 1000  # Convert milliseconds to seconds
    
    print(f"Lunghezza dell'audio: {audio_length_seconds:.2f} secondi")
    _, midi_data, _ = basic_pitch_predict(audio_file, basic_pitch_model)
    midi_file = 'output/test_recorded_audio.mid'
    # Measure length of midi_file in seconds
    midi = PrettyMIDI(midi_file)
    midi_length_seconds = midi.get_end_time()
    print(f"Lunghezza del MIDI: {midi_length_seconds:.2f} secondi")

    midi_data.write(midi_file)
    sequence = tokenizer.encode(midi_file)
    end_time = time.time()
    midi_conversion_time = end_time - start_time

    try:
        prediction_queue.put((sequence.ids, audio_length_seconds), block=False)
    except Exception as e:
        print('Error while inserting midi in queue', e)

    if DEBUG:
        print(f"Midi conversion took {midi_conversion_time} seconds.")
        print(f'Generated midi file: {midi_file}')


def predict(prediction_queue, playback_queue):
    device = get_device()
    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())

    (converted_midi, audio_length_seconds) = prediction_queue.get()
    if converted_midi is not None:
        input_sequence = converted_midi
        start_time = time.time()
        input_sequence = input_sequence
        prediction = compute_output_sequence(model, device, tokenizer, input_sequence, max_seq_len=max_seq_len, max_output_tokens=OUTPUT_TOKENS_COUNT)
        end_time = time.time()
        prediction_time = end_time - start_time
        score = tokenizer.decode(prediction)
        score.dump_midi('output/test_completed_midi.mid')

        playback_queue.put(('output/test_completed_midi.mid', audio_length_seconds), block=False)

        if DEBUG:
            print(f'Prediction took {prediction_time} seconds.', end='\r')

def playback(playback_queue):

    prediction_message = playback_queue.get()

    (prediction_file, audio_length_seconds) = prediction_message

    #if len(prediction) != 0:
    print(f'Generating audio from prediction...')
    start_time = time.time()
    #create_midi_from_sequence(tokenizer, prediction, output_midi_file)
    try:
        audio = PrettyMIDI(prediction_file)
        audio_data = audio.synthesize(fs=OUTPUT_SAMPLE_RATE)
        wavfile.write('output/test_completed_audio_wav.wav', SAMPLE_RATE, audio_data)
        end_time = time.time()
        audio_synthesis_time = end_time - start_time

        audiosegment = AudioSegment.from_wav('output/test_completed_audio_wav.wav')
        new_audio_length_seconds_2 = len(audiosegment) / 1000  # Convert milliseconds to seconds
        
        new_audio_length_seconds = len(audio_data) / OUTPUT_SAMPLE_RATE
        print(f'Lunghezza dell\'audio originale: {audio_length_seconds:.2f} secondi')
        print(f"Lunghezza dell'audio generato: {new_audio_length_seconds:.2f} secondi {audio.get_end_time()} midi end time, {new_audio_length_seconds_2} with audio_segment")
        samples_to_keep = int((new_audio_length_seconds - audio_length_seconds) * OUTPUT_SAMPLE_RATE)
        print(samples_to_keep)
        audio_data = audio_data[-samples_to_keep:]
        if DEBUG:
            print(f"Audio synthesis time: {audio_synthesis_time:.2f} seconds")
        print(audio_data.shape)
        remaining_length_seconds = len(audio_data) / OUTPUT_SAMPLE_RATE
        print(f"Lunghezza rimanente dell'audio dopo la rimozione dei campioni: {remaining_length_seconds:.2f} secondi")
        
        if DEBUG:
            print(f"Playing generated audio...")
        if remaining_length_seconds > 0:
            wavfile.write('output/test_completed_audio_wav_short.wav', SAMPLE_RATE, audio_data)
            sd.play(audio_data, SAMPLE_RATE, blocking=True)

    except Exception as e:
        print('Error while playing audio', e)

def main():

    conversion_queue = Queue(maxsize=1)
    prediction_queue = Queue(maxsize=1)
    playback_queue = Queue(maxsize=1)
    
    converter_thread = Process(target=convert_audio_to_midi, args=(conversion_queue, prediction_queue), name='converter')
    converter_thread.daemon = True
    converter_thread.start()

    predictor_thread = Process(target=predict, args=(prediction_queue, playback_queue), name='predictor')
    predictor_thread.daemon = True
    predictor_thread.start()

    playback_thread = Process(target=playback, args=(playback_queue,), name='playback')
    playback_thread.daemon = True
    playback_thread.start()

    converter_thread.join()
    predictor_thread.join()
    playback_thread.join()

if __name__ == '__main__':
    main()