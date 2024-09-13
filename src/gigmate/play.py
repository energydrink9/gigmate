import io
import os
import time
from typing import Callable, Optional
import requests
from gigmate.utils.audio_utils import convert_audio_to_int_16
import sounddevice as sd
from pretty_midi import PrettyMIDI
from multiprocessing import Process, Queue
import numpy as np
from pydub import AudioSegment

#PREDICTION_HOST = 'https://kib693pk4z7dei-8000.proxy.runpod.net'
PREDICTION_HOST = 'http://localhost:8000'
PREDICTION_URL = PREDICTION_HOST + '/complete-audio'
CHANNELS = 1
SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 22050
BUFFER_SIZE_IN_SECONDS = 12
MINIMUM_AUDIO_BUFFER_LENGTH_IN_SECONDS = 5
MIC_PLUS_SPEAKER_LATENCY_IN_MILLISECONDS = 225 # Use audio_delay_measurement.py to estimate
OUTPUT_BLOCK_SIZE = int(OUTPUT_SAMPLE_RATE / 10)
OUTPUT_PLAYBACK_DELAY = OUTPUT_BLOCK_SIZE / OUTPUT_SAMPLE_RATE
MAX_OUTPUT_LENGTH_IN_SECONDS = 20
MAX_OUTPUT_TOKENS_COUNT = 100
SOUNDFONT_PATH = 'output/Roland SOUNDCanvas SC-55 Up.sf2'# Downloaded from https://archive.org/download/free-soundfonts-sf2-2019-04
MIDI_PROGRAM = None# https://wiki.musink.net/doku.php/midi/instrument
DEBUG = True

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

        audio_file = io.BytesIO()
        audio_segment = AudioSegment(
            audio_buffer.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=CHANNELS
        )
        audio_segment.export(audio_file, format='ogg', codec='libvorbis')

        try:
            audio_buffer_length_in_seconds = get_audio_buffer_length_in_seconds(audio_buffer)
    
            if audio_buffer_length_in_seconds >= MINIMUM_AUDIO_BUFFER_LENGTH_IN_SECONDS:                
                empty_queue(conversion_queue)
                conversion_queue.put((audio_file, record_end_time), block=False)

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

def predict_loop(conversion_queue: Queue, playback_queue: Queue) -> None:
    while True:
        (audio_buffer, record_end_time) = conversion_queue.get()
        
        try:
            files = {'request': audio_buffer}
            data = {
                'max_output_length_in_seconds': MAX_OUTPUT_LENGTH_IN_SECONDS,
                'max_output_tokens_count': MAX_OUTPUT_TOKENS_COUNT,
                'midi_program': MIDI_PROGRAM
            }
            start_time = time.perf_counter()
            response = requests.post(PREDICTION_URL, data=data, files=files)
            converted_midi = io.BytesIO(response.content)
            converted_midi_length = PrettyMIDI(converted_midi).get_end_time()
            end_time = time.perf_counter()
            prediction_file = io.BytesIO(response.content)
            print(f'Completion time: {end_time - start_time:.2f}')
            empty_queue(playback_queue)
            playback_queue.put((prediction_file, record_end_time, converted_midi_length), block=False)    

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
    predicted_midi.write('output/prova123.mid')
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
    playback_queue = Queue(maxsize=1)
    
    listener = Process(target=listen_loop, args=(conversion_queue,), name='listen')
    listener.daemon = True
    listener.start()

    predictor = Process(target=predict_loop, args=(conversion_queue, playback_queue), name='predictor')
    predictor.daemon = True
    predictor.start()

    player = Process(target=playback_loop, args=(playback_queue,), name='playback')
    player.daemon = True
    player.start()

    listener.join()
    predictor.join()
    player.join()

if __name__ == '__main__':
    main()