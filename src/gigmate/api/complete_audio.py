import asyncio
import multiprocessing
import reactivex.scheduler
import time
import traceback
from typing import Any, Callable, Dict, Generic, TypeVar, cast
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import asynccontextmanager
import soundfile as sf
import uvicorn
import reactivex
import reactivex.operators as ops
import numpy as np
from dataclasses import dataclass
import torch
import tempfile

from gigmate.api.latest_concat_map import latest_concat_map
from gigmate.domain.prediction import complete_audio, complete_sequence
from gigmate.model.model import get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.utils.device import get_device

CHUNK_DURATION = 100  # Chunk duration in milliseconds
chunk_size = 44100 * CHUNK_DURATION // 1000  # Assuming a sample rate of 44100 Hz
MAX_CHUNKS = 12
MAX_OUTPUT_LENGTH_IN_SECONDS = 8
MAX_OUTPUT_TOKENS = 180
OUTPUT_SAMPLE_RATE = 22050
INPUT_SAMPLE_RATE = 22050
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


def complete_audio_loop(prediction_queue: multiprocessing.Queue, synthesis_queue: multiprocessing.Queue) -> None:
    device = get_device()
    device = device

    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())

    while True:
        chunk, temperature = prediction_queue.get()
        if chunk is None:
            break
        
        tensor = torch.from_numpy(chunk.data).unsqueeze(0)
        output_wav, output_sr = complete_audio(
            model,
            device,
            tensor,
            INPUT_SAMPLE_RATE,
            max_output_length_in_seconds=MAX_OUTPUT_LENGTH_IN_SECONDS,
            temperature=temperature,
        )
        synthesis_queue.put(AudioChunk(output_wav, chunk.record_start_time, chunk.record_end_time))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources and store them in the app state
 
    app.state.prediction_queue = prediction_queue
    app.state.synthesis_queue = synthesis_queue

    yield  # This will allow the application to run

    # Cleanup code can be added here if needed


app = FastAPI(lifespan=lifespan)


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


@app.websocket("/ws/complete-audio")  # Define the WebSocket route
async def audio_websocket(websocket: WebSocket):

    await websocket.accept()

    # Create an observable for processing audio chunks
    subject: reactivex.Subject[AudioChunk[np.ndarray]] = reactivex.subject.Subject()

    def complete_sequence_step(chunk: AudioChunk[np.ndarray]) -> AudioChunk[np.ndarray]:
        app.state.prediction_queue.put((chunk, DEFAULT_TEMPERATURE))
        return app.state.synthesis_queue.get()
    
    chunk_observable: reactivex.Observable = subject.pipe(
        ops.buffer_with_count(MAX_CHUNKS, 4),
        ops.map(merge_chunks),
        latest_concat_map(lambda chunk: reactivex.from_callable(lambda: complete_sequence_step(chunk))),
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
        on_next=on_next,  # Complete audio and send data back to the client
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
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Start the server
