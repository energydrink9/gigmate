# In the main function, record audio from the microphone, then, every N milliseconds, convert the audio to a MIDI file using basic pitch.
# Pass the MIDI file to the model, and get a prediction to complete the MIDI file. Play the completed MIDI file.

import io
import os
import time
import requests
from gigmate.audio_utils import generate_random_filename
from gigmate.midi_conversion import convert_wav_to_midi
import sounddevice as sd
from pretty_midi import PrettyMIDI
from multiprocessing import Process, Queue
import numpy as np
from scipy.io import wavfile
import soundfile as sf

CHANNELS = 1
SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 22050
OUTPUT_TOKENS_COUNT = 20
BUFFER_SIZE_IN_SECONDS = 25
#PREDICTION_HOST = 'https://liqyj0y9eogrl7-8000.proxy.runpod.net'
PREDICTION_HOST = 'http://localhost:8000'
PREDICTION_URL = PREDICTION_HOST + '/predict'
MINIMUM_AUDIO_BUFFER_LENGTH_IN_SECONDS = 3
DEBUG = False
MIC_PLUS_SPEAKER_LATENCY_IN_MILLISECONDS = 168
OUTPUT_BLOCK_SIZE = int(OUTPUT_SAMPLE_RATE / 10)
OUTPUT_PLAYBACK_DELAY = OUTPUT_BLOCK_SIZE / OUTPUT_SAMPLE_RATE

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if DEBUG:
            print(f"Function {func.__name__} took {end_time - start_time} seconds")
        return result
    return wrapper

def get_audio_buffer_length_in_seconds(audio_buffer):
    return len(audio_buffer) / SAMPLE_RATE

def empty_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()
        except Exception:
            break

def listen(conversion_queue, incoming_audio):
    
    def callback(indata, frames, audio_time, status):
        nonlocal incoming_audio
        record_end_time = time.time()
        sample_time = record_end_time - (audio_time.currentTime - audio_time.inputBufferAdcTime)
        if incoming_audio is None:
            incoming_audio = {
                'audio_buffer': [indata.copy()],
                'audio_start_time': [sample_time],
                'record_end_time': [record_end_time]
            }
        else:
            incoming_audio['audio_buffer'].append(indata.copy())
            incoming_audio['audio_start_time'].append(sample_time)
            incoming_audio['record_end_time'].append(record_end_time)

        audio_buffer_oldness = time.time() - incoming_audio['audio_start_time'][0]

        if DEBUG:
            print(f"Audio buffer oldness: {audio_buffer_oldness}, len {len(incoming_audio['audio_buffer'])}")
        
        while audio_buffer_oldness > BUFFER_SIZE_IN_SECONDS:
            incoming_audio['audio_buffer'] = incoming_audio['audio_buffer'][1:]
            incoming_audio['audio_start_time'] = incoming_audio['audio_start_time'][1:]
            incoming_audio['record_end_time'] = incoming_audio['record_end_time'][1:]
            audio_buffer_oldness = time.time() - incoming_audio['audio_start_time'][0]

        audio_buffer = np.concatenate(incoming_audio['audio_buffer'], axis=0).reshape(-1, 1).astype(np.float32)
        record_end_time = incoming_audio['record_end_time'][-1]

        try:
            empty_queue(conversion_queue)
            conversion_queue.put((audio_buffer, record_end_time), block=False)
            time.sleep(0)
        except Exception as e:
            print('Error while inserting audio buffer in queue')
            print(e)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE / 10), channels=CHANNELS, latency='low', callback=callback):
            sd.sleep(10000)
    except Exception as e:
        print(f"Error while listening: {e}")

def listen_loop(conversion_queue):
    incoming_audio = None

    while True:
        # Decrease the block size to reduce latency
        listen(conversion_queue, incoming_audio)
            
def convert_audio_to_midi(audio_buffer):
    audio_file = generate_random_filename(extension='.wav')
    wavfile.write(audio_file, SAMPLE_RATE, audio_buffer)
    midi = convert_wav_to_midi(audio_file)
    os.remove(audio_file)
    return midi

def convert_audio_to_midi_loop(conversion_queue, prediction_queue):

    while True:
        (audio_buffer, record_end_time) = conversion_queue.get()
        audio_buffer_length_in_seconds = get_audio_buffer_length_in_seconds(audio_buffer)
    
        if audio_buffer_length_in_seconds < MINIMUM_AUDIO_BUFFER_LENGTH_IN_SECONDS:
            continue
    
        converted_midi = convert_audio_to_midi(audio_buffer)
        converted_midi_length = converted_midi.get_end_time()
        in_memory_file = io.BytesIO()
        converted_midi.write(in_memory_file)

        try:
            empty_queue(prediction_queue)
            prediction_queue.put((in_memory_file, record_end_time, converted_midi_length), block=False)
            time.sleep(0)
        except Exception as e:
            print('Error while inserting midi in queue', e)

def predict(converted_midi):
    midi_file = converted_midi.getvalue()
    files = {'request': midi_file}
    try:
        response = requests.post(PREDICTION_URL, files=files, data={ 'output_tokens_count': OUTPUT_TOKENS_COUNT })
        prediction_file = io.BytesIO(response.content)
        return prediction_file

    except Exception as e:
        print(f"Error while calling prediction API: {e}")

def predict_loop(prediction_queue, playback_queue):

    while True:
        converted_midi, record_end_time, converted_midi_length = prediction_queue.get()
        prediction_file = predict(converted_midi)
        try:
            empty_queue(playback_queue)
            playback_queue.put((prediction_file, record_end_time, converted_midi_length), block=False)
        except Exception as e:
            print('Error while inserting in playback queue', e)

        time.sleep(0)

def convert_to_int_16(audio_data):
    max_16bit = 2**15
    raw_data = audio_data * max_16bit
    raw_data = raw_data.astype(np.int16)
    return raw_data

def get_processing_time(record_end_time, current_time):
    return current_time - record_end_time

def get_audio_to_play(prediction_file, record_end_time: float, input_length: float, sample_rate=OUTPUT_SAMPLE_RATE, playback_delay: float = 0, get_current_time=lambda: time.time()):
    
    audio = PrettyMIDI(prediction_file)
    audio_data = audio.fluidsynth(fs=sample_rate)
    processing_time = get_processing_time(record_end_time, get_current_time())

    playback_start_time = input_length + processing_time - playback_delay
    input_samples_to_remove = int(playback_start_time * OUTPUT_SAMPLE_RATE)
    cut_audio_data = audio_data[input_samples_to_remove:]
    remaining_length_seconds = len(cut_audio_data) / OUTPUT_SAMPLE_RATE

#    if DEBUG:
    audio_length_seconds = len(audio_data) / OUTPUT_SAMPLE_RATE
    print(f"Generated audio length: {audio_length_seconds:.2f} seconds")
    print(f'Input length to remove: {input_length:.2f} seconds')
    print(f'Processing time to remove: {processing_time:.2f} seconds')
    print(f'Playback delay: {playback_delay:.2f} seconds')
    print(f"Remaining length after samples removal: {remaining_length_seconds:.2f} seconds")

    if remaining_length_seconds > 0:
        return cut_audio_data

    return None


def playback_loop(playback_queue):
    audio_buffer = None
    audio_index = 0

    def callback(outdata, frames, time, status):
        nonlocal audio_buffer, audio_index
        if status:
            print(status)
        
        if audio_buffer is None:
            outdata[:] = 0
            return

        remaining = len(audio_buffer) - audio_index
        if remaining > 0:
            n = min(remaining, frames)
            outdata[:n, 0] = audio_buffer[audio_index:audio_index+n]
            audio_index += n
            if n < frames:
                outdata[n:] = 0
                audio_buffer = None
                audio_index = 0
        else:
            outdata[:] = 0
            audio_buffer = None
            audio_index = 0

    with sd.OutputStream(samplerate=OUTPUT_SAMPLE_RATE, channels=CHANNELS, callback=callback, blocksize=OUTPUT_BLOCK_SIZE, latency='low'):

        playback_delay = OUTPUT_PLAYBACK_DELAY + MIC_PLUS_SPEAKER_LATENCY_IN_MILLISECONDS / 1000

        while True:
            prediction_message = playback_queue.get()

            try:
                (prediction, record_end_time, converted_midi_length) = prediction_message
                generated_audio = get_audio_to_play(prediction, record_end_time, converted_midi_length, sample_rate=OUTPUT_SAMPLE_RATE, playback_delay=playback_delay)
                if generated_audio is not None:
                    print('Playing audio...')
                    audio_buffer = generated_audio
                    audio_index = 0
                    sd.sleep(int(len(audio_buffer) / OUTPUT_SAMPLE_RATE * 1000))
            
            except Exception as e:
                print('Error while playing audio', e)

def main():

    conversion_queue = Queue(maxsize=1)
    prediction_queue = Queue(maxsize=1)
    playback_queue = Queue(maxsize=1)
    
    listen_thread = Process(target=listen_loop, args=(conversion_queue,), name='listen')
    listen_thread.daemon = True
    listen_thread.start()

    converter_thread = Process(target=convert_audio_to_midi_loop, args=(conversion_queue, prediction_queue), name='converter')
    converter_thread.daemon = True
    converter_thread.start()

    predictor_thread = Process(target=predict_loop, args=(prediction_queue, playback_queue), name='predictor')
    predictor_thread.daemon = True
    predictor_thread.start()

    playback_thread = Process(target=playback_loop, args=(playback_queue,), name='playback')
    playback_thread.daemon = True
    playback_thread.start()

    listen_thread.join()
    converter_thread.join()
    predictor_thread.join()
    playback_thread.join()

if __name__ == '__main__':
    main()