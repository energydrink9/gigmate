import time
from typing import Optional
import typing
import litserve as ls
from gigmate.domain.midi_conversion import convert_audio_to_midi
from gigmate.utils.audio_utils import generate_random_filename
from gigmate.utils.constants import get_params
from gigmate.utils.device import get_device
from gigmate.model.model import get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.domain.prediction import complete_sequence
from gigmate.model.tokenizer import get_tokenizer
from fastapi import Request, Response

DEFAULT_MAX_OUTPUT_TOKENS_COUNT = 1000
DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS = 10
INCLUDE_INPUT = True

class CompleteAudioAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())
        self.device = get_device()
        self.tokenizer = get_tokenizer()
        self.max_seq_len = get_params()['max_seq_len']
        print(f'Using device: {device}')

    def decode_request(self, request: Request) -> tuple[list[int], int, float, Optional[int]]:
        max_output_tokens_count = int(request['max_output_tokens_count']) if 'max_output_tokens_count' in request else DEFAULT_MAX_OUTPUT_TOKENS_COUNT
        max_output_length_in_seconds = float(request['max_output_length_in_seconds']) if 'max_output_length_in_seconds' in request else DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS
        midi_program = int(request['midi_program']) if 'midi_program' in request else None
        bytes = request['request'].file.read()
        return bytes, max_output_tokens_count, max_output_length_in_seconds, midi_program
    
    def predict(self, input: tuple[typing.Any, int, float, Optional[int]]) -> list:
        audio_data, max_output_tokens_count, max_output_length_in_seconds, midi_program = input
        file_path = generate_random_filename(extension='.ogg')
        with open(file_path, 'wb') as file:
            file.write(audio_data)
            score = convert_audio_to_midi(file_path)
            sequence = self.tokenizer.encode(score).ids
            start_time = time.perf_counter()
            prediction = complete_sequence(self.model, self.device, self.tokenizer, sequence, self.max_seq_len, midi_program=midi_program, include_input=INCLUDE_INPUT, max_output_tokens=max_output_tokens_count, max_output_length_in_seconds=max_output_length_in_seconds, verbose=False)
            end_time = time.perf_counter()
            print(f"Predicted {len(prediction)} in {end_time - start_time:.2f} seconds.")
        
            return prediction

    def encode_response(self, prediction: list[int]) -> Response:
        score = self.tokenizer.decode(prediction)
        output = score.dumps_midi()

        return Response(content=output, media_type="application/midi", status_code=200, headers={"Content-Disposition": "attachment; filename=prediction.mid"})


if __name__ == "__main__":
    api = CompleteAudioAPI()
    server = ls.LitServer(api, accelerator="auto", api_path='/complete-audio')
    server.run(port=8000)