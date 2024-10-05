# random seed
from typing import List
from gigmate.model.codec import EncodedFrame, get_codes, get_encoded_frames, get_codec
from gigmate.utils.constants import get_random_seed
import torch

torch.manual_seed(get_random_seed())
encoded_frames: List[EncodedFrame] = [(torch.randn((1, 8, 50)), None), (torch.randn((1, 8, 50)), None)]

def test_get_codes() -> None:
    codes = get_codes(encoded_frames)
    assert codes.shape == (1, 8, 100), 'codes shape is not correct'

def test_codec() -> None:
    codec = get_codec('cpu')
    