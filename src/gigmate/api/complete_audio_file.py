import time
from fastapi.responses import FileResponse
import litserve as ls
from torch import Tensor
import torchaudio
from gigmate.domain.prediction import complete_sequence
from gigmate.model.codec import encode, decode, get_codec
from gigmate.utils.audio_utils import generate_random_filename
from gigmate.utils.constants import get_pad_token_id
from gigmate.model.model import get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from fastapi import Request, Response
from encodec.utils import save_audio

OUTPUT_LENGTH_IN_SECONDS = 10.
INCLUDE_INPUT = True


class SimpleLitAPI(ls.LitAPI):

    device: str

    def setup(self, device) -> None:
        self.device = device
        compile = False
        self.model = get_model(device=self.device, checkpoint_path=get_latest_model_checkpoint_path(), compile=compile)
        self.codec = get_codec(self.device)

    def decode_request(self, request: Request, **kwargs) -> tuple[Tensor, float, float]:
        output_length_in_seconds = float(request['output_length_in_seconds']) if 'output_length_in_seconds' in request else OUTPUT_LENGTH_IN_SECONDS
        file = request['request'].file
        wav, sr = torchaudio.load(file)
        sequence, sr = encode(wav, sr, device=self.device, add_start_and_end_tokens=False)

        return sequence[0], sr, output_length_in_seconds
    
    def predict(self, input: tuple[Tensor, float, float], **kwargs) -> Tensor:
        x, sr, output_length_in_seconds = input
        start_time = time.perf_counter()
        prediction = complete_sequence(
            self.model,
            self.device,
            x,
            frame_rate=self.codec.config.frame_rate,
            padding_value=get_pad_token_id(),
            max_output_length_in_seconds=output_length_in_seconds,
        )
        end_time = time.perf_counter()
        print(f"Predicted {len(prediction)} in {end_time - start_time:.2f} seconds.")
    
        return prediction

    def encode_response(self, prediction: Tensor, **kwargs) -> Response:
        
        output_audio, sr = decode(prediction, self.device)
        output_file_path = generate_random_filename()
        save_audio(output_audio.detach().cpu(), output_file_path, sample_rate=sr)
        return FileResponse(
            path=output_file_path,
            media_type="audio/wav",
            filename="prediction.wav"
        )
    

if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000)