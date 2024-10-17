from typing import Union
import torch


Device = Union[torch.device, str, int]


def get_device() -> Device:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
