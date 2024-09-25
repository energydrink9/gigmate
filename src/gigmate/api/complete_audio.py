import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import reactivex.scheduler
import os
import time
import traceback
from typing import Any, Callable, Dict, Generic, TypeVar, cast
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import asynccontextmanager
from miditok import MusicTokenizer, TokSequence
import soundfile as sf
from spleeter.separator import Separator
import uvicorn
import reactivex
import reactivex.operators as ops
import numpy as np
from dataclasses import dataclass
from symusic.types import Score
import torch
from pretty_midi import PrettyMIDI
import tempfile

from gigmate.api.latest_concat_map import latest_concat_map
from gigmate.domain.midi_conversion import convert_stems_to_midi, merge_midis
from gigmate.domain.prediction import complete_sequence
from gigmate.model.model import get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.model.tokenizer import get_tokenizer
from gigmate.utils.audio_utils import synthesize_midi
from gigmate.utils.constants import get_params
from gigmate.utils.device import get_device

CHUNK_DURATION = 100  # Chunk duration in milliseconds
chunk_size = 44100 * CHUNK_DURATION // 1000  # Assuming a sample rate of 44100 Hz
MAX_CHUNKS = 12
MAX_OUTPUT_LENGTH_IN_SECONDS = 8
MAX_OUTPUT_TOKENS = 180
MIDI_PROGRAM = None #33
OUTPUT_SAMPLE_RATE = 22050
DEFAULT_TEMPERATURE = 0.3

T = TypeVar('T')
U = TypeVar('U')

@dataclass(frozen=True)
class AudioChunk(Generic[T]):
    data: T
    record_start_time: float
    record_end_time: float

prediction_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
synthesis_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)

def complete_midi_loop(prediction_queue: multiprocessing.Queue, synthesis_queue: multiprocessing.Queue) -> None:
    device = get_device()
    device = device

    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())
    tokenizer = get_tokenizer()
    max_seq_len = get_params()['max_seq_len']

    while True:
        chunk, temperature = prediction_queue.get()
        if chunk is None:
            break
        completed_chunk = complete_midi_sequence(chunk, tokenizer, device, model, max_seq_len, temperature=temperature, show_progress=True)
        synthesis_queue.put(completed_chunk)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources and store them in the app state
 
    app.state.separator = Separator('spleeter:5stems')
    app.state.prediction_queue = prediction_queue
    app.state.synthesis_queue = synthesis_queue

    yield  # This will allow the application to run

    # Cleanup code can be added here if needed

app = FastAPI(lifespan=lifespan)

def get_audio_stems(chunk: AudioChunk[np.ndarray], separator: Separator) -> AudioChunk[dict[str, np.ndarray]]:
    start_time = time.perf_counter()
    stems = separator.separate(chunk.data)
    end_time = time.perf_counter()
    print(f"1. Separating audio stems took {end_time - start_time} seconds")
    return AudioChunk(stems, chunk.record_start_time, chunk.record_end_time)

def extract_audio_chunk(audio_data, i, chunk_size, start_time) -> AudioChunk[np.ndarray]:
    start = i * chunk_size
    end = (i + 1) * chunk_size
    record_end_time = start_time + (end - start) / 44100
    return AudioChunk(
        data=audio_data[start:end],
        record_start_time=start_time,
        record_end_time=record_end_time
    )

def merge_chunks(chunks: list[AudioChunk[dict[str, np.ndarray]]]) -> AudioChunk[dict[str, np.ndarray]]:
    start_time = time.perf_counter()
    if len(chunks) == 0:
        return AudioChunk(dict(), 0, 0)
    
    data = dict()
    stems = chunks[0].data.keys()
    
    for stem in stems:
        data[stem] = np.concatenate([chunk_data for chunk in chunks for chunk_stem, chunk_data in chunk.data.items() if chunk_stem == stem])

    end_time = time.perf_counter()
    print(f"2. Merging chunks took {end_time - start_time} seconds")
    return AudioChunk(
        data,
        min([chunk.record_start_time for chunk in chunks]),
        max([chunk.record_end_time for chunk in chunks]),
    )

def get_temporary_file(audio: np.ndarray) -> tempfile._TemporaryFileWrapper:
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.wav', delete=False) as file:
        sf.write(file, audio, OUTPUT_SAMPLE_RATE, format='WAV')
        file.flush()  # Ensure the data is written to the file
        file.seek(0)  # Reset the file pointer to the beginning
        return file

def convert_to_midi(stems: AudioChunk[dict[str, np.ndarray]]) -> AudioChunk[dict[str, Score]]:
    start_time = time.perf_counter()
    executor = ThreadPoolExecutor(multiprocessing.cpu_count())
    file_stems = dict({stem[0]: get_temporary_file(stem[1]).name for stem in stems.data.items()})
    midis = convert_stems_to_midi(file_stems, executor)
    for file_name in file_stems.values():
        os.remove(file_name)
    end_time = time.perf_counter()
    print(f"3. Convert_to_midi took {end_time - start_time} seconds")
    return AudioChunk(midis, stems.record_start_time, stems.record_end_time)

def merge_midi_chunks(chunk: AudioChunk[dict[str, Score]]) -> AudioChunk[Score]:
    score = merge_midis(list(chunk.data.items()))
    return AudioChunk(score, chunk.record_start_time, chunk.record_end_time)

def complete_midi_sequence(chunk: AudioChunk[Score], tokenizer: MusicTokenizer, device: str, model: torch.nn.Module, max_seq_len, temperature, show_progress=False) -> AudioChunk[Score]:
    start_time = time.perf_counter()
    token_sequence: TokSequence = cast(TokSequence, tokenizer.encode(chunk.data))
    input_sequence = cast(list[int], token_sequence.ids)
    output_sequence = complete_sequence(model, device, tokenizer, input_sequence, max_seq_len=max_seq_len, max_output_tokens=MAX_OUTPUT_TOKENS, max_output_length_in_seconds=MAX_OUTPUT_LENGTH_IN_SECONDS, temperature=temperature, show_progress=show_progress)
    sequence: Score = tokenizer.decode(output_sequence)
    end_time = time.perf_counter()
    print(f'4. Completed sequence in {end_time - start_time} seconds')
    return AudioChunk(sequence, chunk.record_start_time, chunk.record_end_time)

def synthesize_midi_from_chunk(chunk: AudioChunk[Score]) -> AudioChunk[np.ndarray]:
    start_time = time.perf_counter()
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.mid', delete=True) as temp_file:
        score = chunk.data
        score.dump_midi(temp_file.name)
        midi = PrettyMIDI(temp_file.name)
        data = synthesize_midi(midi, MIDI_PROGRAM, OUTPUT_SAMPLE_RATE)
        # not let's cut the audio data removing (chunk.record_end_time - chunk.record_start_time) at the beginning:
        samples_to_remove = int((chunk.record_end_time - chunk.record_start_time) * OUTPUT_SAMPLE_RATE)
        # Remove the samples from the beginning of the audio data
        data = data[samples_to_remove:]
        end_time = time.perf_counter()
        print(f'5. Synthesized midi in {end_time - start_time} seconds')
        return AudioChunk(data, chunk.record_start_time, chunk.record_end_time)

@app.websocket("/ws/complete-audio")  # Define the WebSocket route
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    
    # Access resources from app.state
    state = websocket.app.state
    separator = state.separator

    # Create an observable for processing audio chunks
    subject: reactivex.Subject[AudioChunk[np.ndarray]] = reactivex.subject.Subject()

    def get_audio_stems_step(chunk: AudioChunk[np.ndarray]) -> AudioChunk[dict[str, np.ndarray]]:
        return get_audio_stems(chunk, separator=separator)
    
    def from_iterable(chunk: AudioChunk[dict[str, np.ndarray]]) -> reactivex.Observable[AudioChunk[dict[str, np.ndarray]]]:
        return reactivex.from_iterable([chunk])
    
    def from_callable(callable: Callable[[], U]) -> reactivex.Observable[U]:
        return reactivex.from_callable(callable)

    def complete_midi_sequence_step(chunk: AudioChunk[Score]) -> AudioChunk[Score]:
        app.state.prediction_queue.put((chunk, DEFAULT_TEMPERATURE))
        return app.state.synthesis_queue.get()

    chunk_observable: reactivex.Observable = subject.pipe(
        latest_concat_map(lambda chunk: reactivex.from_callable(lambda: get_audio_stems_step(chunk))),
        ops.buffer_with_count(MAX_CHUNKS, 4),
        ops.map(merge_chunks),
        latest_concat_map(lambda chunk: reactivex.from_callable(lambda: convert_to_midi(chunk))),
        ops.map(merge_midi_chunks),
        latest_concat_map(lambda chunk: reactivex.from_callable(lambda: complete_midi_sequence_step(chunk))),
        ops.map(synthesize_midi_from_chunk),
    )

    def get_json_response(chunk: AudioChunk) -> Dict[str, Any]:
        return {
            "startTime": chunk.record_start_time,
            "endTime": chunk.record_end_time,
            "data": chunk.data.tolist()
        }

    async def send_chunk(chunk: AudioChunk, websocket: WebSocket) -> None:
        await websocket.send_json(get_json_response(chunk))

    loop = asyncio.get_running_loop()

    def on_next(chunk: AudioChunk) -> None:
        loop.create_task(send_chunk(chunk, websocket))

    def on_error(error: Exception) -> None:
        print(f"Error: {traceback.print_exception(error)}")

    # Subscribe to the observable
    subscription = chunk_observable.subscribe(
        on_next=on_next,  # Send MIDI data back to the client
        on_error=on_error,  # Handle errors
        scheduler=reactivex.scheduler.ThreadPoolScheduler(max_workers=multiprocessing.cpu_count())
    )

    def get_chunk(data):
        audio_data = np.array(data['audio'], dtype=np.float32)
        record_start_time = data['startTime']
        record_end_time = data['endTime']
        return AudioChunk(audio_data, record_start_time, record_end_time)

    try:
        while True:
            data = await websocket.receive_json()  # Expecting a JSON object
            chunk = get_chunk(data)
            subject.on_next(chunk)  # Push chunk to the subject

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        subscription.dispose()  # Dispose of the subscription when done
        subject.dispose()


if __name__ == "__main__":  # Server initialization
    complete_midi_sequence_process = multiprocessing.Process(target=complete_midi_loop, args=(prediction_queue, synthesis_queue), name='prediction_process')
    complete_midi_sequence_process.daemon = True
    complete_midi_sequence_process.start()

    uvicorn.run(app, host="0.0.0.0", port=8000)  # Start the server

    complete_midi_sequence_process.join()
