# random seed
from typing import List
from gigmate.model.codec import EncodedFrame, get_codes, get_encoded_frames, model
from gigmate.utils.constants import get_random_seed
import torch

torch.manual_seed(get_random_seed())
encoded_frames: List[EncodedFrame] = [(torch.randn((1, 8, 50)), None), (torch.randn((1, 8, 50)), None)]

def test_get_codes() -> None:
    codes = get_codes(encoded_frames)
    assert codes.shape == (1, 8, 100), 'codes shape is not correct'

def test_get_encoded_frames() -> None:
    codes = get_codes(encoded_frames)
    encoded_frames_reconstructed = get_encoded_frames(codes, model.frame_rate)
    assert encoded_frames_reconstructed[0][0].shape == (1, 8, 100), 'wrong reconstructed encoded frames shape'
    assert torch.equal(encoded_frames_reconstructed[0][0], torch.cat([encoded_frames[0][0], encoded_frames[1][0]], dim=-1)), 'encoded frames are not correct'

