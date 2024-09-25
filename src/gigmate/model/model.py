from typing import Any, Callable, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from gigmate.utils.constants import get_params
from gigmate.model.cached_multihead_attention import CachedMultiheadAttention, look_ahead_mask

MODEL_FILE_NAME = 'output/model.chk'
ENABLE_TORCHSCRIPT = True

class TransformerBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, dff, max_seq_len: int, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.mha = CachedMultiheadAttention(d_model, num_heads, max_seq_len=max_seq_len)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(dropout)

    @torch.jit.export
    def reset_kv_cache(self):
        self.mha.reset_kv_cache()

    def forward(self, x, attn_mask, key_padding_mask: Optional[torch.Tensor], use_cache: bool = False, current_token_index: Optional[int] = None):

        attn_output, _ = self.mha(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=True, use_cache=use_cache, current_token_index=current_token_index)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_seq_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        def get_transformer_block():
            return TransformerBlock(d_model, num_heads, dff, max_seq_len, dropout)
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)  # Use padding_idx for masking

        transformers = [get_transformer_block() for _ in range(num_layers)]
        self.transformer_layers = nn.ModuleList(transformers)
        self.dense = nn.Linear(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding1D(d_model)
        self.max_seq_len = max_seq_len
    
    @torch.jit.export
    def reset_kv_cache(self):
        for transformer in self.transformer_layers.children():
            transformer.reset_kv_cache()
            
    def forward(self, input, use_cache: bool = False, current_token_index: Optional[int] = None):
        x = self.embedding(input)
        
        # Add positional encoding
        # TODO: Fix positional encoding for sequences longer than max_seq_len, for example using relative positional encoding or updating the positional encodings of items in the cache
        pos_encoding = self.pos_encoding
        x = x + pos_encoding(x)

        if use_cache and current_token_index is not None:
            x = x[:, current_token_index:current_token_index + 1, :]

        sequence_length = x.size(1)

        attn_mask: Optional[torch.Tensor] = None
        key_padding_mask: Optional[torch.Tensor] = None

        if use_cache and current_token_index is not None:
        # when we pass kvcache, we are passing single token as input which need to attend to all previous tokens, but not the tokens after it, the size of the mask should be equal to max seq len
            attn_mask = look_ahead_mask(self.max_seq_len)[current_token_index:current_token_index + 1, :].to(x.device)
            key_padding_mask = look_ahead_mask(self.max_seq_len)[current_token_index:current_token_index + 1, :].to(x.device)
        else:
            attn_mask = look_ahead_mask(sequence_length).to(x.device)

        # Pass through transformer blocks
        for layer in self.transformer_layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, use_cache=use_cache, current_token_index=current_token_index)

        # Final output layer
        x = self.dense(x)

        return x

# Workaround to load the model
def load_ckpt(model, checkpoint_path: str, device: str):
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace("model._orig_mod.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)

def get_model(params = get_params(), checkpoint_path = None, device: str = 'cpu') -> Union[nn.Module, Callable[..., Any]]:
    model = TransformerModel(
        params['num_layers'],
        params['d_model'],
        params['num_heads'],
        params['dff'],
        params['vocab_size'],
        params['max_seq_len'],
        params['dropout_rate']
    )
    if ENABLE_TORCHSCRIPT:
        model = torch.jit.script(model)

    model.to(device)

    if checkpoint_path is not None:
        load_ckpt(model, checkpoint_path, device)

    if device == 'cuda':
        torch.set_float32_matmul_precision('high')
        return torch.compile(model)

    return model

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding1D(nn.Module):
    
    cached_penc: Optional[torch.Tensor]

    def __init__(self, channels: int):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor: torch.Tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        cached_penc = self.cached_penc
        if cached_penc is not None and cached_penc.shape == tensor.shape:
            return cached_penc

        cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)

        self.cached_penc = cached_penc

        return cached_penc
