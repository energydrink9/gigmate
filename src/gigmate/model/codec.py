from transformers import EncodecModel, AutoProcessor
from encodec.utils import convert_audio
import torchaudio
import torch
import typing

EncodedFrame = typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]

def get_codes(encoded_frames: typing.List[EncodedFrame]) -> torch.Tensor:
    return torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

def get_encoded_frames(codes: torch.Tensor, frame_rate: int) -> typing.List[EncodedFrame]:
    #return [(code, None) for code in codes.split(frame_rate, dim=-1)]
    return [(codes, None)]

def get_codec(device: str):
    model = EncodecModel.from_pretrained("facebook/encodec_32khz", normalize=False).to(device)
    print(model.config)
    return model.eval()

def get_processor():
    return AutoProcessor.from_pretrained("facebook/encodec_32khz")

def encode(audio_path: str, device: str) -> tuple[torch.Tensor, float]:

    model = get_codec(device)
    processor = get_processor()

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, processor.sampling_rate, model.config.audio_channels)
    wav = wav[:,:(processor.sampling_rate * 2)]

    inputs = processor(raw_audio=wav[0], sampling_rate=processor.sampling_rate, return_tensors="pt")
 
    bandwidth = 2.2
    output = model.encode(inputs["input_values"].to(device), inputs["padding_mask"].to(device), bandwidth=bandwidth)

    return output.audio_codes[0], model.config.frame_rate

def decode(codes: torch.Tensor, device: str) -> torch.Tensor:
    model = get_codec(device)
    decoded_wav = model.decode(codes.unsqueeze(0).to(device), [None])
    output_tensor = decoded_wav['audio_values'].squeeze(0).detach().cpu()
    return output_tensor