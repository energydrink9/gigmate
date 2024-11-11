from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from gigmate.model.cached_multihead_attention import CachedMultiheadAttention


class TransformerBlock(nn.Module):
    layernorm2: Union[nn.LayerNorm, nn.Identity]
    dropout2: Union[nn.Dropout, nn.Identity]
    
    def __init__(self, d_model: int, num_heads: int, dff: int, sliding_window_size: int, dropout: float = 0.1, has_cross_attention: bool = False):
        super(TransformerBlock, self).__init__()

        self.has_cross_attention = has_cross_attention
        self.self_attention = CachedMultiheadAttention(d_model, num_heads, sliding_window_size=sliding_window_size)
        
        if self.has_cross_attention is True:
            self.cross_attention = CachedMultiheadAttention(d_model, num_heads, sliding_window_size=sliding_window_size)
            self.layernorm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.layernorm2 = nn.Identity()
            self.dropout2 = nn.Identity()
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
            self,
            x: Tensor,
            cross_attention_src: Optional[Tensor] = None,
            sequence_lengths: Optional[List[int]] = None,
            cross_attention_sequence_lengths: Optional[List[int]] = None,
            use_cache: bool = False,
            cache: Optional[Tensor] = None,
            cache_index: Optional[int] = None,
            encoder: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        norm1 = self.layernorm1(x)
        if torch.isnan(norm1).any():
            print('Warning: nan in norm1')

        tgt2, cache = self.self_attention(
            norm1,
            norm1,
            norm1,
            use_cache=use_cache,
            cache=cache,
            cache_index=cache_index,
            sequence_lengths=sequence_lengths,
            kv_sequence_lengths=sequence_lengths,
            inverted_query=encoder,
            inverted_key_values=encoder,
        )
        if torch.isnan(tgt2).any():
            print('Warning: nan in self attention')

        x = x + self.dropout1(tgt2)
        if torch.isnan(x).any():
            print('Warning: nan in dropout 1')

        if self.has_cross_attention is True:
            tgt2 = self.layernorm2(x)
            if torch.isnan(tgt2).any():
                print('Warning: nan in layernorm2')
            if cross_attention_src is not None and torch.isnan(cross_attention_src).any():
                print('Warning: nan in cross_attention_src')
            tgt2, cross_attention_cache = self.cross_attention(
                tgt2,
                cross_attention_src,
                cross_attention_src,
                use_cache=False,
                sequence_lengths=sequence_lengths,
                kv_sequence_lengths=cross_attention_sequence_lengths,
                inverted_key_values=True,
                causal=False,
            )
            if torch.isnan(tgt2).any():
                print('Warning: nan in cross attention')
            
            x = x + self.dropout2(tgt2)
            if torch.isnan(x).any():
                print('Warning: nan in dropout 2')

        tgt2 = self.layernorm3(x)
        if torch.isnan(tgt2).any():
            print('Warning: nan in norm 3')
        tgt2 = self.ffn(tgt2)
        if torch.isnan(tgt2).any():
            print('Warning: nan in ffn')
        x = x + self.dropout3(tgt2)
        if torch.isnan(x).any():
            print('Warning: nan in dropout 3')

        return x, cache
