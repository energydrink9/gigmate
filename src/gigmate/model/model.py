import math
from typing import List, Optional, Tuple, cast
import torch
import torch.nn as nn
import torchao
from torch import Tensor
from gigmate.dataset.dataset import SequenceLengths
from gigmate.model.decoder import Decoder
from gigmate.model.encoder import Encoder
from gigmate.model.utils import compile_model
from gigmate.utils.constants import USE_SLIDING_WINDOW, get_pad_token_id, get_params, get_use_alibi, get_use_custom_model
from gigmate.utils.device import Device


ENABLE_QUANTIZATION = False
USE_ALIBI = get_use_alibi()
USE_CUSTOM_MODEL = get_use_custom_model()

# 12. check out grouped query attention
# 15. Implement inference


def positional_encoding(seq_len, d_model, dtype=torch.float):
    position = torch.arange(seq_len, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=dtype) * -(math.log(10000.0) / d_model))
    
    pos_encoding = torch.zeros(1, seq_len, d_model)
    pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
    pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding


# Helper function for initializing weights. Ref: https://www.yadavsaurabh.com/good-initialisation-of/
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Kaiming initialization for linear layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if hasattr(m, 'special_init'):
            nn.init.constant_(m.weight, 0)


def get_key_padding_mask(max_seq_len: int, sequence_lengths: List[int], device=None, inverted: bool = False) -> Tensor:
    batch_size = len(sequence_lengths)
    kpm = torch.full((batch_size, max_seq_len), False, device=device)

    if inverted is True:
        for i in range(batch_size):
            sequence_length = sequence_lengths[i]
            kpm[i, :max_seq_len - sequence_length] = True
    else:
        for i in range(batch_size):
            sequence_length = sequence_lengths[i]
            kpm[i, sequence_length:] = True

    return kpm


def generate_causal_sliding_mask(
    sz: int,
    sliding_window_size: Optional[int],
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if sz < 1:
        raise ValueError("Size must be positive")

    mask = torch.triu(
        torch.ones((sz, sz), dtype=torch.bool, device=device),
        diagonal=1,
    )
    
    if sliding_window_size is not None:
        if sliding_window_size < 1:
            raise ValueError("Sliding window size must be positive")

        # If sliding window is smaller than sequence length,
        # also mask out tokens too far in the past
        if sliding_window_size < sz:
            window_mask = torch.tril(
                torch.ones((sz, sz), dtype=torch.bool, device=device),
                diagonal=-sliding_window_size
            )
            mask = mask | window_mask
    
    return mask


class TransformerModel(nn.Module):
    def __init__(
        self,
        encoder_layers: int,
        decoder_layers: int,
        d_model: int,
        codebooks: int,
        num_heads: int,
        dff: int,
        vocab_size: int,
        max_decoder_seq_len: int,
        max_seq_len: int,
        sliding_window_size: Optional[int],
        dropout: float = 0.1,
        padding_value=0,
    ):
        
        super(TransformerModel, self).__init__()

        self.num_heads = num_heads
        self.codebooks = codebooks
        self.num_layers = decoder_layers
        self.sliding_window_size = sliding_window_size

        embeddings = [nn.Embedding(vocab_size, d_model, padding_idx=padding_value) for _ in range(codebooks)]
        self.embeddings = nn.ModuleList(embeddings)

        if USE_CUSTOM_MODEL:
            self.encoder = Encoder(encoder_layers, d_model, num_heads, dff, sliding_window_size, dropout)
            self.decoder = Decoder(decoder_layers, d_model, num_heads, dff, sliding_window_size, dropout)
        else:
            self.transformer = torch.nn.Transformer(
                d_model,
                num_heads,
                encoder_layers,
                decoder_layers,
                dff,
                dropout,
                activation='gelu',
                norm_first=True,
                batch_first=True,
            )
        
        linears = [nn.Linear(d_model, vocab_size) for _ in range(codebooks)]
        self.linears = nn.ModuleList(linears)

        self.max_decoder_seq_len = max_decoder_seq_len
        self.max_seq_len = max_seq_len
        self.pos_encoding = positional_encoding(max_seq_len + max_decoder_seq_len, d_model)

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
    ) -> Tuple[Tensor, Optional[List[Tensor]], Optional[Tensor]]:

        x = cast(torch.Tensor, sum([self.embeddings[k](input[:, k]) for k in range(self.codebooks)]))
        full_track_x = cast(torch.Tensor, sum([self.embeddings[k](conditioning_input[:, k]) for k in range(self.codebooks)]))
        
        if not USE_ALIBI:
            # Add positional encoding
            x = x + self.pos_encoding[:, self.max_seq_len:self.max_seq_len + self.max_decoder_seq_len, :].to(x.device)
            full_track_x = full_track_x + self.pos_encoding[:, :self.max_seq_len, :].to(full_track_x.device)

        if USE_CUSTOM_MODEL:
            # TODO: Implement kv cache for the encoder
            if encoder_cache is None:
                cross_attention_src, _ = self.encoder(
                    full_track_x,
                    sequence_lengths.full_track if sequence_lengths is not None else None,
                )
                
                # Only the last sliding window is used for cross attention
                if self.sliding_window_size is not None:
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

        else:
            
            tgt_mask = generate_causal_sliding_mask(self.max_decoder_seq_len, self.sliding_window_size, device=x.device)
            x = self.transformer(
                full_track_x,
                x,
                tgt_mask=tgt_mask,
                src_key_padding_mask=get_key_padding_mask(self.max_seq_len, sequence_lengths.full_track, device=full_track_x.device, inverted=True) if sequence_lengths is not None else None,
                tgt_key_padding_mask=get_key_padding_mask(self.max_decoder_seq_len, sequence_lengths.stem, device=x.device) if sequence_lengths is not None else None,
                src_is_causal=False,
                tgt_is_causal=True,
            )
            updated_cache = None
            cross_attention_src = None

        logits = torch.stack([self.linears[k](x) for k in range(self.codebooks)], dim=1)  # [B, K, S, vocab_size]

        return logits, updated_cache, cross_attention_src


def load_ckpt(model, checkpoint_path: str, device: Device) -> None:
    print(f'Loading ckpt {checkpoint_path}')
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)['state_dict']
    
    # Workaround to load the model
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("_orig_mod.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=True)


def get_model(params=get_params(), checkpoint_path=None, device: Device = 'cpu', compile: bool = True) -> TransformerModel:

    model = TransformerModel(
        encoder_layers=params['encoder_layers'],
        decoder_layers=params['decoder_layers'],
        d_model=params['d_model'],
        codebooks=params['codebooks'],
        num_heads=params['num_heads'],
        dff=params['dff'],
        vocab_size=params['vocab_size'],
        max_decoder_seq_len=params['max_decoder_seq_len'],
        max_seq_len=params['max_seq_len'],
        sliding_window_size=params['sliding_window_size'] if USE_SLIDING_WINDOW else None,
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
