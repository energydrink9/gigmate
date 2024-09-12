import os
import time
import litserve as ls
from gigmate.domain.midi_conversion import convert_audio_to_midi
from gigmate.utils.audio_utils import generate_random_filename
from fastapi import Request, Response
from symusic import Score

DEFAULT_MAX_OUTPUT_TOKENS_COUNT = 1000
DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS = 10
INCLUDE_INPUT = True

class ConvertToMidiAPI(ls.LitAPI):
    def setup(self, devices):
        pass

    def decode_request(self, request: Request) -> str:
        bytes = request['request'].file.read()

        return bytes
    
    def predict(self, file_content) -> list:
        try:
            file_path = generate_random_filename(extension='.mp3')
            with open(file_path, 'wb') as file:
                file.write(file_content)

            start_time = time.perf_counter()
            score = convert_audio_to_midi(file_path)
            end_time = time.perf_counter()
            print(f"Converted audio to midi in {end_time - start_time:.2f} seconds.")
            return score
        
        finally:
            os.remove(file_path) # cleanup temporary file

    def encode_response(self, score: Score) -> Response:
        output = score.dumps_midi()
        return Response(content=output, media_type="application/midi", status_code=200, headers={"Content-Disposition": "attachment; filename=converted.mid"})
    

if __name__ == "__main__":
    api = ConvertToMidiAPI()
    server = ls.LitServer(api, accelerator="auto", api_path='/convert-to-midi')
    server.run(port=8002)