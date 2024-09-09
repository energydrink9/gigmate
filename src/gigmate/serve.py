import time
from typing import Optional
import litserve as ls
import symusic
from gigmate.constants import get_params
from gigmate.device import get_device
from gigmate.model import get_model
from gigmate.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.predict import compute_output_sequence
from gigmate.tokenizer import get_tokenizer
from fastapi import Request, Response

DEFAULT_MAX_OUTPUT_TOKENS_COUNT = 1000
DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS = 10
INCLUDE_INPUT = True

class SimpleLitAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())
        self.device = get_device()
        self.tokenizer = get_tokenizer()
        self.max_seq_len = get_params()['max_seq_len']

    def decode_request(self, request: Request) -> tuple[list[int], int, float, Optional[int]]:
        max_output_tokens_count = int(request['max_output_tokens_count']) if 'max_output_tokens_count' in request else DEFAULT_MAX_OUTPUT_TOKENS_COUNT
        max_output_length_in_seconds = float(request['max_output_length_in_seconds']) if 'max_output_length_in_seconds' in request else DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS
        midi_program = int(request['midi_program']) if 'midi_program' in request else None
        bytes = request['request'].file.read()
        score = symusic.Score.from_midi(bytes)
        sequence = self.tokenizer.encode(score)
        return sequence.ids, max_output_tokens_count, max_output_length_in_seconds, midi_program
    
    def predict(self, input: tuple[list[int], int, float, Optional[int]]) -> list:
        x, max_output_tokens_count, max_output_length_in_seconds, midi_program = input
        start_time = time.perf_counter()
        prediction = compute_output_sequence(self.model, self.device, self.tokenizer, x, self.max_seq_len, midi_program=midi_program, include_input=INCLUDE_INPUT, max_output_tokens=max_output_tokens_count, max_output_length_in_seconds=max_output_length_in_seconds, verbose=False)
        end_time = time.perf_counter()
        print(f"Predicted {len(prediction)} in {end_time - start_time:.2f} seconds.")
    
        return prediction

    def encode_response(self, prediction: list[int]) -> Response:
        score = self.tokenizer.decode(prediction)
        output = score.dumps_midi()

        return Response(content=output, media_type="application/midi", status_code=200, headers={"Content-Disposition": "attachment; filename=prediction.mid"})


if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000)