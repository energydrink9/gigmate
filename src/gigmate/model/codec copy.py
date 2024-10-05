from encodec.model import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import torch
import typing

EncodedFrame = typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]

model = EncodecModel.encodec_model_24khz()
# The number of codebooks used will be determined by the bandwidth selected.
# E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
model.set_target_bandwidth(6.0)

def get_codes(encoded_frames: typing.List[EncodedFrame]) -> torch.Tensor:
    return torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

def get_encoded_frames(codes: torch.Tensor, frame_rate: int) -> typing.List[EncodedFrame]:
    #return [(code, None) for code in codes.split(frame_rate, dim=-1)]
    return [(codes, None)]

def encode(audio_path: str) -> tuple[torch.Tensor, float]:

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)

    codes = get_codes(encoded_frames)
    
    return codes, model.frame_rate

# Write the decode function:
def decode(codes: torch.Tensor) -> torch.Tensor:
    # Compute the quantized discrete latent vectors
    with torch.no_grad():
        encoded_frames = get_encoded_frames(codes, model.frame_rate)
        for f in encoded_frames:
            print(f[0].shape)
        decoded_wav = model.decode(encoded_frames)
    
    return decoded_wav