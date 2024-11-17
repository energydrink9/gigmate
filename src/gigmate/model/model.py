from typing import List, Optional, Tuple, cast
import torch
import torch.nn as nn
import torchao
from torch import Tensor
from gigmate.dataset.dataset import SequenceLengths
from gigmate.model.decoder import Decoder
from gigmate.model.encoder import Encoder
from gigmate.model.utils import compile_model
from gigmate.utils.constants import get_pad_token_id, get_params
from gigmate.utils.device import Device

ENABLE_QUANTIZATION = False

# 12. check out grouped query attention
# 15. Implement inference


# Helper function for initializing weights. Ref: https://www.yadavsaurabh.com/good-initialisation-of/
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Kaiming initialization for linear layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if hasattr(m, 'special_init'):
            nn.init.constant_(m.weight, 0)


class TransformerModel(nn.Module):
    def __init__(self, encoder_layers: int, decoder_layers: int, d_model: int, codebooks: int, num_heads: int, dff: int, vocab_size: int, sliding_window_size: int, dropout: float = 0.1, padding_value=0):
        super(TransformerModel, self).__init__()

        self.num_heads = num_heads
        self.codebooks = codebooks
        self.num_layers = decoder_layers
        self.sliding_window_size = sliding_window_size

        embeddings = [nn.Embedding(vocab_size, d_model, padding_idx=padding_value) for _ in range(codebooks)]
        self.embeddings = nn.ModuleList(embeddings)

        self.encoder = Encoder(encoder_layers, d_model, num_heads, dff, sliding_window_size, dropout)
        self.decoder = Decoder(decoder_layers, d_model, num_heads, dff, sliding_window_size, dropout)
        
        linears = [nn.Linear(d_model, vocab_size) for _ in range(codebooks)]
        self.linears = nn.ModuleList(linears)

        self.apply(init_weights)
    
    def forward(
            self,
            input: Tensor,
            conditioning_input: Tensor,
            sequence_lengths: Optional[SequenceLengths] = None,
            use_cache: bool = False,
            cache: Optional[List[Tensor]] = None,
            cache_index: Optional[int] = None,
            encoder_cache: Optional[Tensor] = None
    ) -> Tuple[Tensor, List[Tensor], Tensor]:

        x = cast(torch.Tensor, sum([self.embeddings[k](input[:, k]) for k in range(self.codebooks)]))

        # TODO: Implement kv cache for the encoder
        if encoder_cache is None:
            full_track_x = sum([self.embeddings[k](conditioning_input[:, k]) for k in range(self.codebooks)])
            cross_attention_src, _ = self.encoder(
                full_track_x,
                sequence_lengths.full_track if sequence_lengths is not None else None,
            )
            
            # Only the last sliding window is used for cross attention
            cross_attention_src = cross_attention_src[:, -self.sliding_window_size:, :]

        else:
            cross_attention_src = encoder_cache

        x, updated_cache = self.decoder(
            x,
            sequence_lengths=sequence_lengths.stem if sequence_lengths is not None else None,
            cross_attention_sequence_lengths=sequence_lengths.full_track if sequence_lengths is not None else None,
            cross_attention_src=cross_attention_src,
            use_cache=use_cache,
            cache=cache,
            cache_index=cache_index,
        )

        logits = torch.stack([self.linears[k](x) for k in range(self.codebooks)], dim=1)  # [B, K, S, vocab_size]

        return logits, updated_cache, cross_attention_src


def load_ckpt(model, checkpoint_path: str, device: Device) -> None:
    print(f'Loading ckpt {checkpoint_path}')
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)['state_dict']
    
    # Workaround to load the model
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("_orig_mod.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)  # TODO: set strict back to True


def get_model(params=get_params(), checkpoint_path=None, device: Device = 'cpu', compile=True) -> TransformerModel:

    model = TransformerModel(
        encoder_layers=params['encoder_layers'],
        decoder_layers=params['decoder_layers'],
        d_model=params['d_model'],
        codebooks=params['codebooks'],
        num_heads=params['num_heads'],
        dff=params['dff'],
        vocab_size=params['vocab_size'],
        sliding_window_size=params['sliding_window_size'],
        dropout=params['dropout_rate'],
        padding_value=get_pad_token_id(),
    )

    model.to(device)

    if checkpoint_path is not None:
        load_ckpt(model, checkpoint_path, device)

    # Compiled flex attention does not work on mps device
    if compile is True:
        model = cast(TransformerModel, compile_model(model, device_type=device))

    if device == 'cuda':
        # Maybe the next line is not needed when quantization is enabled as it would be taken care of by torchao?
        torch.set_float32_matmul_precision('high')        

    if ENABLE_QUANTIZATION is True:
        model = torchao.autoquant(model)

    return model
