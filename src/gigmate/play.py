import io
import os
import time
from typing import Callable, Optional
import requests
from gigmate.utils.audio_utils import convert_audio_to_int_16
import sounddevice as sd
from pretty_midi import PrettyMIDI
from multiprocessing import Process, Queue
from threading import Thread
import numpy as np
from pydub import AudioSegment

CONVERSION_HOST = 'https://sh1qkip4z0etci-8001.proxy.runpod.net'
#CONVERSION_HOST = 'http://localhost:8001'
CONVERSION_URL = CONVERSION_HOST + '/convert-to-midi'
PREDICTION_HOST = 'https://qqu8ody09ec7l4-8000.proxy.runpod.net'
#PREDICTION_HOST = 'http://localhost:8000'
PREDICTION_URL = PREDICTION_HOST + '/complete-midi'
CHANNELS = 1
SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 22050
BUFFER_SIZE_IN_SECONDS = 20
MINIMUM_AUDIO_BUFFER_LENGTH_IN_SECONDS = 5
MIC_PLUS_SPEAKER_LATENCY_IN_MILLISECONDS = 225 # Use audio_delay_measurement.py to estimate
OUTPUT_BLOCK_SIZE = int(OUTPUT_SAMPLE_RATE / 10)
OUTPUT_PLAYBACK_DELAY = OUTPUT_BLOCK_SIZE / OUTPUT_SAMPLE_RATE
MAX_OUTPUT_LENGTH_IN_SECONDS = 10
MAX_OUTPUT_TOKENS_COUNT = 90
SOUNDFONT_PATH = 'output/Roland SOUNDCanvas SC-55 Up.sf2'# Downloaded from https://archive.org/download/free-soundfonts-sf2-2019-04
MIDI_PROGRAM = None# https://wiki.musink.net/doku.php/midi/instrument
DEBUG = False

def measure_time(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if DEBUG:
            print(f"Function {func.__name__} took {end_time - start_time} seconds")
        return result
    return wrapper

def get_audio_buffer_length_in_seconds(audio_buffer: np.ndarray) -> float:
    return len(audio_buffer) / SAMPLE_RATE

def empty_queue(queue: Queue) -> None:
    while not queue.empty():
        try:
            queue.get_nowait()
        except Exception:
            break

def listen(conversion_queue: Queue, incoming_audio: Optional[dict]) -> None:
    
    def callback(indata, frames, audio_time, status):
        nonlocal incoming_audio

        # TODO: The current time might not be accurate because there is no guarantee on when the callback will be invoked.
        # use a more accurate value for the record end time.
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

        record_end_time = incoming_audio['record_end_time'][-1]
        audio_buffer = convert_audio_to_int_16(np.concatenate(incoming_audio['audio_buffer'], axis=0))

        mp3_file = io.BytesIO()
        audio_segment = AudioSegment(
            audio_buffer.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=CHANNELS
        )
        audio_segment.export(mp3_file, format='mp3')

        try:
            audio_buffer_length_in_seconds = get_audio_buffer_length_in_seconds(audio_buffer)
    
            if audio_buffer_length_in_seconds >= MINIMUM_AUDIO_BUFFER_LENGTH_IN_SECONDS:                
                empty_queue(conversion_queue)
                conversion_queue.put((mp3_file, record_end_time), block=False)

        except Exception as e:
            print('Error while inserting audio buffer in conversion queue')
            print(e)

    try:
        # Decrease the block size to reduce latency
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE / 10), channels=CHANNELS, latency='low', callback=callback):
            while True:
                sd.sleep(10000)
    except Exception as e:
        print(f"Error while listening: {e}")

def listen_loop(conversion_queue: Queue) -> None:
    incoming_audio = None

    listen(conversion_queue, incoming_audio)
            
# def convert_audio_to_midi(audio_buffer: np.ndarray) -> PrettyMIDI:
#     audio_file = generate_random_filename(extension='.wav')
#     wavfile.write(audio_file, SAMPLE_RATE, audio_buffer)
#     midi = convert_wav_to_midi(audio_file)
#     os.remove(audio_file)
#     return midi

# def convert_audio_to_midi_loop(conversion_queue: Queue, prediction_queue: Queue) -> None:

#     while True:
#         (audio_buffer, record_end_time) = conversion_queue.get()
#         audio_buffer_length_in_seconds = get_audio_buffer_length_in_seconds(audio_buffer)
    
#         if audio_buffer_length_in_seconds < MINIMUM_AUDIO_BUFFER_LENGTH_IN_SECONDS:
#             continue
    
#         score = convert_audio_to_midi(audio_buffer)
#         converted_midi_length = calculate_score_length_in_seconds(score)
#         in_memory_file = io.BytesIO()
#         score.dump_midi(in_memory_file)

#         try:
#             empty_queue(prediction_queue)
#             prediction_queue.put((in_memory_file, record_end_time, converted_midi_length), block=False)
#             time.sleep(0)
#         except Exception as e:
#             print('Error while inserting midi in prediction queue', e)

def convert_audio_to_midi_loop(conversion_queue: Queue, prediction_queue: Queue) -> None:

    while True:
        (audio_buffer, record_end_time) = conversion_queue.get()

        try:
            files = {'request': audio_buffer}
            start_time = time.perf_counter()
            response = requests.post(CONVERSION_URL, files=files)
            end_time = time.perf_counter()
            print(f'Conversion time: {end_time - start_time:.2f}')
            in_memory_file = io.BytesIO(response.content)
            converted_midi_length = PrettyMIDI(in_memory_file).get_end_time()

            try:
                empty_queue(prediction_queue)
                prediction_queue.put((in_memory_file, record_end_time, converted_midi_length), block=False)
                time.sleep(0)
            except Exception as e:
                print('Error while inserting midi in prediction queue', e)

        except Exception as e:
            print(f"Error while calling conversion API: {e}")

def predict(converted_midi: io.BytesIO, max_output_length_in_seconds: float, max_output_tokens_count: int, midi_program: int) -> io.BytesIO:
    midi_file = converted_midi.getvalue()
    files = {'request': midi_file}
    try:
        data = {
            'max_output_length_in_seconds': max_output_length_in_seconds,
            'max_output_tokens_count': max_output_tokens_count,
            'midi_program': midi_program
        }
        start_time = time.perf_counter()
        response = requests.post(PREDICTION_URL, files=files, data=data)
        end_time = time.perf_counter()
        print(f'Inference time: {end_time - start_time:.2f}')
        return io.BytesIO(response.content)

    except Exception as e:
        print(f"Error while calling prediction API: {e}")

def predict_thread(converted_midi, record_end_time, converted_midi_length, playback_queue):
    prediction_file = predict(converted_midi, MAX_OUTPUT_LENGTH_IN_SECONDS, MAX_OUTPUT_TOKENS_COUNT, MIDI_PROGRAM)
    empty_queue(playback_queue)
    playback_queue.put((prediction_file, record_end_time, converted_midi_length), block=False)    

def predict_loop(prediction_queue: Queue, playback_queue: Queue) -> None:

    while True:
        
        try:
            predict_threads = []
            for _ in range(2):
                converted_midi, record_end_time, converted_midi_length = prediction_queue.get()
                thread = Thread(target=predict_thread, args=(converted_midi, record_end_time, converted_midi_length, playback_queue))
                thread.daemon = True
                thread.start()
                predict_threads.append(thread)
                time.sleep(2)

            for thread in predict_threads:
                thread.join()

        except Exception as e:
            print('Error while performing prediction loop', e)

        time.sleep(0)

def get_processing_time(record_end_time: float, current_time: float) -> float:
    return current_time - record_end_time

def get_program_midi(predicted_midi: PrettyMIDI, midi_program: int) -> PrettyMIDI:
    filtered_midi = PrettyMIDI()

    instrument_code = midi_program if midi_program != -1 else 0
    
    print(f'Predicted instruments: {predicted_midi.instruments}')
    
    # Iterate through all instruments in the predicted MIDI
    for instrument in predicted_midi.instruments:
        if instrument.program == instrument_code:
            # Add the instrument to the filtered MIDI
            filtered_midi.instruments.append(instrument)

    return filtered_midi

def get_audio_to_play(
    prediction_file: io.BytesIO,
    record_end_time: float,
    input_length: float,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    playback_delay: float = 0,
    get_current_time: Callable[[], float] = lambda: time.time()
) -> Optional[np.ndarray]:
    
    predicted_midi = PrettyMIDI(prediction_file)
    predicted_program_midi = get_program_midi(predicted_midi, MIDI_PROGRAM) if MIDI_PROGRAM is not None else predicted_midi
    sf2_path = SOUNDFONT_PATH if os.path.exists(SOUNDFONT_PATH) else None
    audio_data = predicted_program_midi.fluidsynth(fs=sample_rate, sf2_path=sf2_path)
    processing_time = get_processing_time(record_end_time, get_current_time())

    playback_start_time = input_length + processing_time + playback_delay
    input_samples_to_remove = int(playback_start_time * OUTPUT_SAMPLE_RATE)
    cut_audio_data = audio_data[input_samples_to_remove:]
    remaining_length_seconds = len(cut_audio_data) / OUTPUT_SAMPLE_RATE

    if DEBUG:
        audio_length_seconds = len(audio_data) / OUTPUT_SAMPLE_RATE
        print(f"Generated audio length: {audio_length_seconds:.2f} seconds")
        print(f'Input length to remove: {input_length:.2f} seconds')
        print(f'Processing time to remove: {processing_time:.2f} seconds')
        print(f'Playback delay: {playback_delay:.2f} seconds')
        print(f"Remaining length after samples removal: {remaining_length_seconds:.2f} seconds")

    if remaining_length_seconds > 0:
        return cut_audio_data

    return None


def playback_loop(playback_queue: Queue) -> None:
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

def main() -> None:

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