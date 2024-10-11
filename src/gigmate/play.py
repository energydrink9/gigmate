import json
from multiprocessing import Queue
import time
from typing import Callable, Optional
import websocket
import sounddevice as sd
from threading import Thread
import numpy as np

# SERVER_HOST = 'xze3ji413brpy4-8000.proxy.runpod.net'
SERVER_HOST = 'localhost:8000'
SERVER_URL = 'ws://' + SERVER_HOST + '/ws/complete-audio'
CHANNELS = 1
SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 22050
MIC_PLUS_SPEAKER_LATENCY_IN_MILLISECONDS = 225  # Use audio_delay_measurement.py to estimate
OUTPUT_BLOCK_SIZE = int(OUTPUT_SAMPLE_RATE / 10)
OUTPUT_PLAYBACK_DELAY = OUTPUT_BLOCK_SIZE / OUTPUT_SAMPLE_RATE
MAX_OUTPUT_LENGTH_IN_SECONDS = 8
MAX_OUTPUT_TOKENS_COUNT = 180
MIDI_PROGRAM = None  # https://wiki.musink.net/doku.php/midi/instrument
DEBUG = True


def listen(ws: websocket.WebSocketApp) -> None:
    
    def callback(indata, frames, audio_time, status):

        # TODO: The current time might not be accurate because there is no guarantee on when the callback will be invoked.
        # use a more accurate value for the record end time.
        record_end_time = time.time()

        sample_time = record_end_time - (audio_time.currentTime - audio_time.inputBufferAdcTime)

        # Create the payload
        payload = {
            'audio': indata.copy().tolist(),
            'startTime': sample_time,
            'endTime': record_end_time
        }

        # Send the JSON payload
        ws.send(json.dumps(payload))

    try:
        # Decrease the block size to reduce latency
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=int((SAMPLE_RATE / 10) * 5), channels=CHANNELS, latency='low', callback=callback):
            while True:
                sd.sleep(10000)
    except Exception as e:
        print(f"Error while listening: {e}")


def get_processing_time(record_end_time: float, current_time: float) -> float:
    return current_time - record_end_time


def get_audio_to_play(
    audio_data: np.ndarray,
    record_end_time: float,
    playback_delay: float = 0,
    get_current_time: Callable[[], float] = lambda: time.time()
) -> Optional[np.ndarray]:
    
    processing_time = get_processing_time(record_end_time, get_current_time())
    playback_start_time = processing_time + playback_delay
    input_samples_to_remove = int(playback_start_time * OUTPUT_SAMPLE_RATE)
    cut_audio_data = audio_data[input_samples_to_remove:]
    remaining_length_seconds = len(cut_audio_data) / OUTPUT_SAMPLE_RATE

    if DEBUG:
        audio_length_seconds = len(audio_data) / OUTPUT_SAMPLE_RATE
        print(f"Generated audio length: {audio_length_seconds:.2f} seconds")
        print(f'Processing time to remove: {processing_time:.2f} seconds')
        print(f'Playback delay: {playback_delay:.2f} seconds')
        print(f"Remaining length after samples removal: {remaining_length_seconds:.2f} seconds")

    if remaining_length_seconds > 0:
        return cut_audio_data

    return None


def playback(playback_queue: Queue) -> None:
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
            outdata[:n, 0] = audio_buffer[audio_index: audio_index + n]
            audio_index += n
            if n < frames:
                outdata[n:] = 0
                audio_buffer = None
                audio_index = 0
        else:
            outdata[:] = 0
            audio_buffer = None
            audio_index = 0

    playback_delay = OUTPUT_PLAYBACK_DELAY + MIC_PLUS_SPEAKER_LATENCY_IN_MILLISECONDS / 1000

    def on_message(message: str, sd):
        nonlocal audio_buffer, audio_index
        # Parse the incoming message
        data = json.loads(message)
        audio_data = np.array(data['data'], dtype=np.float32)  # Assuming audio is in this format
        # record_start_time = data['startTime']
        record_end_time = data['endTime']

        print(f'Received audio with length: {len(audio_data) / OUTPUT_SAMPLE_RATE:.2f} seconds')
        print(f'Time elapsed since end of recording: {time.time() - record_end_time}')

        generated_audio = get_audio_to_play(audio_data, record_end_time, playback_delay=playback_delay)
        if generated_audio is not None:
            print('Playing audio...')
            audio_buffer = generated_audio
            audio_index = 0
            sd.sleep(int(len(audio_buffer) / OUTPUT_SAMPLE_RATE * 1000))

    with sd.OutputStream(samplerate=OUTPUT_SAMPLE_RATE, channels=CHANNELS, callback=callback, blocksize=OUTPUT_BLOCK_SIZE, latency='low'):
        while True:
            message = playback_queue.get()
            on_message(message, sd)
            sd.sleep(100)


def on_open(ws: websocket.WebSocket):
    print('Opened websocket')


def on_error(ws: websocket.WebSocket, error):
    print("Error:", error)


def on_close(ws: websocket.WebSocket, _, _2):
    print("Connection closed")


def on_message(message, playback_queue: Queue):
    playback_queue.put(message)


def main() -> None:
    playback_queue: Queue = Queue()

    ws = websocket.WebSocketApp(
        SERVER_URL,
        on_open=on_open,
        on_error=on_error,
        on_close=on_close,
        on_message=lambda _ws, msg: on_message(msg, playback_queue)
    )

    player = Thread(target=playback, args=([playback_queue]), name='playback')
    player.daemon = True
    player.start()

    listener = Thread(target=listen, args=(ws,), name='listen')
    listener.daemon = True
    listener.start()

    ws.run_forever()

    listener.join()
    player.join()


if __name__ == '__main__':
    main()