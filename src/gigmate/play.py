# In the main function, record audio from the microphone, then, every N milliseconds, convert the audio to a MIDI file using basic pitch.
# Pass the MIDI file to the model, and get a prediction to complete the MIDI file. Play the completed MIDI file.

import time
from gigmate.model import get_latest_model_checkpoint
import sounddevice as sd
import torch
from basic_pitch.inference import predict as basic_pitch_predict, Model
from basic_pitch import build_icassp_2022_model_path, FilenameSuffix
from gigmate.predict import compute_output_sequence, create_midi_from_sequence, get_tokenizer, get_params, get_device
from pretty_midi import PrettyMIDI
from multiprocessing import Process, Queue
import numpy as np
from scipy.io import wavfile

CHANNELS = 1
SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 44100
OUTPUT_TOKENS_COUNT = 5
BUFFER_SIZE_IN_SECONDS = 10
DEBUG = False

tokenizer = get_tokenizer()
max_seq_len = get_params()['max_seq_len']
basic_pitch_model = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))

def get_audio_buffer_length_in_seconds(audio_buffer):
    return len(audio_buffer) / SAMPLE_RATE

def empty_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()
        except Exception:
            break

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

        empty_queue(conversion_queue)
        audio_buffer = np.concatenate(incoming_audio['audio_buffer'], axis=0).reshape(-1, 1).astype(np.float32)
        audio_start_time = incoming_audio['audio_start_time'][0]

        conversion_queue.put((audio_buffer, audio_start_time), block=False)
        
    while True:
        # Decrease the block size to reduce latency
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE / 10), channels=CHANNELS, latency='low', callback=callback):
            sd.sleep(10000)

def convert_audio_to_midi(conversion_queue, prediction_queue):

    while True:
        try:
            (audio_buffer, time_zero) = conversion_queue.get()
            audio_buffer_length_in_seconds = get_audio_buffer_length_in_seconds(audio_buffer)
            
            if audio_buffer_length_in_seconds < 0.1:
                continue

            start_time = time.time()
            # Write audio buffer to file
            audio_file = 'output/recorded_audio.wav'
            wavfile.write(audio_file, SAMPLE_RATE, audio_buffer)
            _, midi_data, _ = basic_pitch_predict(audio_file, basic_pitch_model)
            midi_file = 'output/recorded_audio.mid'
            midi_data.write(midi_file)
            sequence = tokenizer.encode(midi_file)
            end_time = time.time()
            midi_conversion_time = end_time - start_time

            empty_queue(prediction_queue)
            prediction_queue.put((sequence.ids, time_zero), block=False)

            if DEBUG:
                print(f"Midi conversion took {midi_conversion_time} seconds.")

        except Exception as e:
            print('Error', e)

def predict(prediction_queue, playback_queue):
    device = get_device()
    model = get_latest_model_checkpoint(device)

    while True:
        converted_midi = prediction_queue.get()
        if converted_midi is not None:
            (input_sequence, time_zero) = converted_midi
            start_time = time.time()
            input_sequence = torch.tensor(input_sequence).to(device)
            prediction = compute_output_sequence(model, tokenizer, input_sequence, max_seq_len, output_tokens=OUTPUT_TOKENS_COUNT)
            end_time = time.time()
            prediction_time = end_time - start_time

            empty_queue(playback_queue)
            playback_queue.put((prediction, time_zero), block=False)

            if DEBUG:
                print(f'Prediction took {prediction_time} seconds.', end='\r')

def playback(playback_queue):

    while True:
        prediction_message = playback_queue.get()

        (prediction, time_zero) = prediction_message

        if len(prediction) != 0:
            print(f'Generating audio from prediction {prediction}...')
            start_time = time.time()
            output_midi_file = 'output/completed_midi.mid'
            create_midi_from_sequence(tokenizer, prediction, output_midi_file)
            audio = PrettyMIDI(output_midi_file)
            audio_data = audio.synthesize(fs=OUTPUT_SAMPLE_RATE)
            wavfile.write('output/completed_audio_wav.wav', SAMPLE_RATE, audio_data)
            end_time = time.time()
            audio_synthesis_time = end_time - start_time
            playback_time = time.time() - time_zero

            print(audio_data.shape)

            samples_to_remove = int(playback_time * OUTPUT_SAMPLE_RATE)
            print(samples_to_remove)
            audio_length_seconds = len(audio_data) / OUTPUT_SAMPLE_RATE
            print(f'Playback time: {playback_time:.2f} seconds')
            print(f"Lunghezza dell'audio generato: {audio_length_seconds:.2f} secondi")
            if DEBUG:
                print(f"Audio synthesis time: {audio_synthesis_time:.2f} seconds")
            audio_data = audio_data[samples_to_remove:]
            wavfile.write('output/completed_audio_wav_short.wav', SAMPLE_RATE, audio_data)
            print(audio_data.shape)
            remaining_length_seconds = len(audio_data) / OUTPUT_SAMPLE_RATE
            print(f"Lunghezza rimanente dell'audio dopo la rimozione dei campioni: {remaining_length_seconds:.2f} secondi")
            
            if DEBUG:
                print(f"Playing generated audio...")
            if remaining_length_seconds > 0:
                sd.play(audio_data, SAMPLE_RATE, blocking=True)

def main():

    conversion_queue = Queue(maxsize=1)
    prediction_queue = Queue(maxsize=1)
    playback_queue = Queue(maxsize=1)
    
    listen_thread = Process(target=listen, args=(conversion_queue,), name='listen')
    listen_thread.daemon = True
    listen_thread.start()

    converter_thread = Process(target=convert_audio_to_midi, args=(conversion_queue, prediction_queue), name='converter')
    converter_thread.daemon = True
    converter_thread.start()

    predictor_thread = Process(target=predict, args=(prediction_queue, playback_queue), name='predictor')
    predictor_thread.daemon = True
    predictor_thread.start()

    playback_thread = Process(target=playback, args=(playback_queue,), name='playback')
    playback_thread.daemon = True
    playback_thread.start()

    listen_thread.join()
    converter_thread.join()
    predictor_thread.join()
    playback_thread.join()

if __name__ == '__main__':
    main()