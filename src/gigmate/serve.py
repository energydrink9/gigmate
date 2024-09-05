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
from fastapi import UploadFile, Response

DEFAULT_OUTPUT_TOKENS_COUNT = 40

class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        # Setup the model so it can be called in `predict`.
        self.model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())
        self.device = get_device()
        self.tokenizer = get_tokenizer()
        self.max_seq_len = get_params()['max_seq_len']

    def decode_request(self, request: UploadFile):
        bytes = request.file.read()
        score = symusic.Score.from_midi(bytes)
        sequence = self.tokenizer.encode(score)
        
        return sequence.ids
    
    def predict(self, x):
        start_time = time.time()
        # Run the model on the input and return the output.
        input_sequence = torch.tensor(x).to(self.device)
        prediction = compute_output_sequence(self.model, self.tokenizer, input_sequence, self.max_seq_len, output_tokens=DEFAULT_OUTPUT_TOKENS_COUNT)
        end_time = time.time()
        print(f"Predicted {len(prediction)} in {end_time - start_time} seconds.")
    
        return prediction

    def encode_response(self, prediction):
        # Convert the model output to a response payload.
        score = self.tokenizer.decode(prediction)
        output = score.dumps_midi()

        return Response(content=output, media_type="application/midi", status_code=200, headers={"Content-Disposition": "attachment; filename=prediction.mid"})


# STEP 2: START THE SERVER
if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000)