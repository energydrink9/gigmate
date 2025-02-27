from typing import List, Optional, Tuple, Union
import torch.nn as nn
from torch import Tensor
from gigmate.model.cached_multihead_attention import CachedMultiheadAttention


class TransformerBlock(nn.Module):
    layernorm2: Union[nn.LayerNorm, nn.Identity]
    dropout2: Union[nn.Dropout, nn.Identity]
    
    def __init__(self, d_model: int, num_heads: int, dff: int, sliding_window_size: Optional[int], dropout: float = 0.1, has_cross_attention: bool = False, start_token: int = 0):
        super(TransformerBlock, self).__init__()

        self.has_cross_attention = has_cross_attention
        self.self_attention = CachedMultiheadAttention(d_model, num_heads, sliding_window_size=sliding_window_size)
        
        if self.has_cross_attention is True:
            self.cross_attention = CachedMultiheadAttention(d_model, num_heads, sliding_window_size=sliding_window_size, start_token=start_token, is_cross_attention=True)
            self.layernorm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.layernorm2 = nn.Identity()
            self.dropout2 = nn.Identity()
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
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

        norm_1 = self.layernorm1(x)

        attn_1, cache = self.self_attention(
            norm_1,
            norm_1,
            norm_1,
            use_cache=use_cache,
            cache=cache,
            cache_index=cache_index,
            sequence_lengths=sequence_lengths,
            kv_sequence_lengths=sequence_lengths,
            inverted_query=encoder,
            inverted_key_values=encoder,
        )

        x = x + self.dropout1(attn_1)

        if self.has_cross_attention is True:
            norm_2 = self.layernorm2(x)
            attn_2, cross_attention_cache = self.cross_attention(
                norm_2,
                cross_attention_src,
                cross_attention_src,
                use_cache=False,
                sequence_lengths=sequence_lengths,
                kv_sequence_lengths=cross_attention_sequence_lengths,
                inverted_key_values=True,
            )
            
            x = x + self.dropout2(attn_2)

        norm_3 = self.layernorm3(x)
        ffn_output = self.ffn(norm_3)
        x = x + self.dropout3(ffn_output)

        return x, cache
