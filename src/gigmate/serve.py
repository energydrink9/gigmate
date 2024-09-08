import time
import litserve as ls
import symusic
import torch
from gigmate.constants import get_params
from gigmate.device import get_device
from gigmate.model import get_model
from gigmate.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.predict import compute_output_sequence
from gigmate.tokenizer import get_tokenizer
from fastapi import Request, Response

DEFAULT_OUTPUT_TOKENS_COUNT = 40
INCLUDE_INPUT = False

class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())
        self.device = get_device()
        self.tokenizer = get_tokenizer()
        self.max_seq_len = get_params()['max_seq_len']

    def decode_request(self, request: Request):
        output_tokens_count = int(request['output_tokens_count']) if 'output_tokens_count' in request else DEFAULT_OUTPUT_TOKENS_COUNT
        bytes = request['request'].file.read()
        score = symusic.Score.from_midi(bytes)
        sequence = self.tokenizer.encode(score)
        return sequence.ids, output_tokens_count
    
    def predict(self, input):
        x, output_tokens_count = input
        start_time = time.perf_counter()
        input_sequence = torch.tensor(x).to(self.device)
        prediction = compute_output_sequence(self.model, self.tokenizer, input_sequence, self.max_seq_len, include_input=INCLUDE_INPUT, output_tokens=output_tokens_count)
        end_time = time.perf_counter()
        print(f"Predicted {len(prediction)} in {end_time - start_time:.2f} seconds.")
    
        return prediction

    def encode_response(self, prediction):
        score = self.tokenizer.decode(prediction)
        output = score.dumps_midi()

        return Response(content=output, media_type="application/midi", status_code=200, headers={"Content-Disposition": "attachment; filename=prediction.mid"})


if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000)