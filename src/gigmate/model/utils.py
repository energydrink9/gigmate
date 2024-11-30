from typing import Union
import torch


def compile_model(model: Union[torch.nn.Module, type[torch.nn.Module]], device_type: str, fullgraph=True):
    # inductor does not support mps just yet
    backend = 'aot_eager' if device_type == 'cuda' else 'aot_eager'
    print(f'Compiling model with backend={backend} and fullgraph={fullgraph}')
    return torch.compile(model, fullgraph=fullgraph, backend=backend)  # , mode='max-autotune')