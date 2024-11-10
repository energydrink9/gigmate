from typing import List, Optional, Tuple
import torch.nn as nn
from torch import Tensor
from gigmate.model.transformer_block import TransformerBlock


class Encoder(nn.Module):

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, sliding_window_size: int, dropout: float):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.sliding_window_size = sliding_window_size
        self.dropout = dropout

        transformers = [self.get_transformer_block() for _ in range(num_layers)]
        self.transformer_layers = nn.ModuleList(transformers)

    def get_transformer_block(self) -> TransformerBlock:
        return TransformerBlock(
            self.d_model,
            self.num_heads,
            self.dff,
            sliding_window_size=self.sliding_window_size,
            dropout=self.dropout,
        )

    def forward(self, x: Tensor, sequence_lengths: Optional[Tensor], use_cache: bool = False, cache: Optional[List[Tensor]] = None, cache_index: Optional[int] = None) -> Tuple[Tensor, Optional[List[Tensor]]]:
        updated_cache: List[Tensor] = []

        for i, layer in enumerate(list(self.transformer_layers)):
            layer_cache = cache[i] if use_cache and cache is not None else None
            x, updated_layer_cache = layer(x, sequence_lengths=sequence_lengths, use_cache=use_cache, cache=layer_cache, cache_index=cache_index, encoder=True)
            updated_cache.append(updated_layer_cache)

        # Only the last sliding window length is used for cross attention
        x = x[:, :, -self.sliding_window_size:]
        
        return x, updated_cache