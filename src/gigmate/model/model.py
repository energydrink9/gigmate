from typing import List, Optional, Tuple, cast
import torch
import torch.nn as nn
import torchao
from torch import Tensor
from gigmate.utils.constants import get_pad_token_id, get_params
from gigmate.model.cached_multihead_attention import CachedMultiheadAttention
from gigmate.utils.device import Device

ENABLE_TORCHSCRIPT = False
ENABLE_QUANTIZATION = False

# [1]. Use alibi positional embeddings
# [2]. Use causal local attention
# [3]. Fix torch compile optimization
# [4]. Apply delay interleaving pattern
# [5]. Use Meta Encodec 32khz for encoding to tokens and decoding back to audio
# [6]. Apply quantization aware training: https://github.com/pytorch/ao
# [7]. Add support for stems
# [8]. Handle padding using appropriate attention mask or builting pytorch nested tensors
# [9]. Create music dataset
# # [1]. Find platform
# # [2]. Download songs on online storage
# # [3]. Create metadata consisting in song (multiple versions), (guitar stem)
# # [4]. Create pipeline step to add noise to the song and save separately
# # [5]. Encode everything with EnCodec 32khz
# # [6]. Publish
# # [7]. Write code for the dataset (splitting and loading of song tracks for training, validation and test)
# [10]. Define evaluation metrics (Frechet Audio Distance)
# [11]. Finish flex attention implementation
# 12. check out grouped query attention
# 13. Perform training
# 14. Evaluate model
# 15. Implement inference
# 16. Implement encoder-decoder version


class TransformerBlock(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, dff: int, sliding_window_size: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.mha = CachedMultiheadAttention(d_model, num_heads, sliding_window_size=sliding_window_size)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sequence_lengths: Optional[List[int]] = None, use_cache: bool = False, cache: Optional[Tensor] = None, cache_index: Optional[int] = None) -> Tuple[Tensor, Optional[Tensor]]:
        
        attn_output, cache = self.mha(x, use_cache=use_cache, cache=cache, cache_index=cache_index, sequence_lengths=sequence_lengths)
        torch.isnan(attn_output).any()  # Seems to prevent nans from propagating. TODO: understand way

        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, cache


class TransformerModel(nn.Module):
    def __init__(self, num_layers: int, d_model: int, codebooks: int, num_heads: int, dff: int, vocab_size: int, batch_size: int, sliding_window_size: int, dropout: float = 0.1, padding_value=0):
        super(TransformerModel, self).__init__()

        def get_transformer_block():
            return TransformerBlock(d_model, num_heads, dff, sliding_window_size=sliding_window_size, dropout=dropout)

        self.num_heads = num_heads
        self.codebooks = codebooks
        self.num_layers = num_layers

        embeddings = [nn.Embedding(vocab_size, d_model, padding_idx=padding_value) for _ in range(codebooks)]
        self.embeddings = nn.ModuleList(embeddings)

        transformers = [get_transformer_block() for _ in range(num_layers)]
        self.transformer_layers = nn.ModuleList(transformers)
        
        linears = [nn.Linear(d_model, vocab_size) for _ in range(codebooks)]
        self.linears = nn.ModuleList(linears)
    
    def forward(self, input: Tensor, sequence_lengths: Optional[List[int]] = None, use_cache: bool = False, cache: Optional[List[Tensor]] = None, cache_index: Optional[int] = None) -> Tuple[Tensor, List[Tensor]]:

        x = sum([self.embeddings[k](input[:, k]) for k in range(self.codebooks)])
        updated_cache: List[Tensor] = []

        # Pass through transformer blocks
        for i, layer in enumerate(list(self.transformer_layers)):
            layer_cache = cache[i] if use_cache and cache is not None else None
            x, updated_layer_cache = layer(x, sequence_lengths, use_cache=use_cache, cache=layer_cache, cache_index=cache_index)
            updated_cache.append(updated_layer_cache)

        # Final output layer
        logits = torch.stack([self.linears[k](x) for k in range(self.codebooks)], dim=1)  # [B, K, S, vocab_size]

        return logits, updated_cache


def load_ckpt(model, checkpoint_path: str, device: Device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)['state_dict']
    
    # Workaround to load the model
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "")] = state_dict.pop(key)
    
    model.load_state_dict(state_dict, strict=True)


def get_model(params=get_params(), checkpoint_path=None, device: Device = 'cpu', compile=True) -> TransformerModel:
    model = TransformerModel(
        num_layers=params['num_layers'],
        d_model=params['d_model'],
        codebooks=params['codebooks'],
        num_heads=params['num_heads'],
        dff=params['dff'],
        vocab_size=params['vocab_size'],
        batch_size=params['batch_size'],
        sliding_window_size=params['sliding_window_size'],
        dropout=params['dropout_rate'],
        padding_value=get_pad_token_id(),
    )
    if ENABLE_TORCHSCRIPT:
        model = cast(TransformerModel, torch.jit.script(model))

    model.to(device)

    if checkpoint_path is not None:
        print(f'Loading ckpt {checkpoint_path}')
        load_ckpt(model, checkpoint_path, device)

    if compile is True:
        # TODO: fix torch compile full graph
        backend = 'aot_eager' if device == 'mps' or device == 'cuda' else 'inductor'
        model = cast(TransformerModel, torch.compile(model, fullgraph=False, backend=backend))

    # TODO: enable quantization if on CUDA
    if device == 'cuda':
        # Probably the next line is not needed when quantization is enabled as it would be taken care of by torchao
        torch.set_float32_matmul_precision('high')
        
        if ENABLE_QUANTIZATION is True:
            model = torchao.autoquant(model)

    return model
