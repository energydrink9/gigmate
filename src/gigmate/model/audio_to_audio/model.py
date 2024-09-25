import math
from typing import Any, Callable, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from gigmate.utils.constants import get_params
from gigmate.model.cached_multihead_attention import CachedMultiheadAttention, look_ahead_mask

ENABLE_TORCHSCRIPT = True

# [1]. Use Alibi positional embeddings
# 2. Use Local attention
# 3. Use Meta Encodec for encoding to tokens and decoding back to audio
# 4. Create music dataset
# 5. Perform training
# 6. Evaluate model
# 7. Implement inference

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

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
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int, max_seq_len: int, batch_size: int, dropout: float=0.1):
        super(TransformerModel, self).__init__()
        def get_transformer_block():
            return TransformerBlock(d_model, num_heads, dff, max_seq_len, dropout)
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)  # Use padding_idx for masking

        transformers = [get_transformer_block() for _ in range(num_layers)]
        self.transformer_layers = nn.ModuleList(transformers)
        self.dense = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
            else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
 

        self.slopes = torch.Tensor(get_slopes(num_heads))
        #In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in the paper). 
        #If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
        #This works because the softmax operation is invariant to translation, and our bias functions are always linear. 
        self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(num_heads, -1, -1)
        self.alibi = self.alibi.view(num_heads, 1, max_seq_len)
        self.alibi = self.alibi.repeat(batch_size//max_seq_len, 1, 1)  # batch_size, 1, 1
    
    def buffered_future_mask(self, tensor):
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(1) < self.args.tokens_per_sample
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([self.args.tokens_per_sample, self.args.tokens_per_sample])), 1
            )
            self._future_mask = self._future_mask.unsqueeze(0) + self.alibi
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:tensor.shape[0]*self.args.decoder_attention_heads, :dim, :dim]

    @torch.jit.export
    def reset_kv_cache(self):
        for transformer in self.transformer_layers.children():
            transformer.reset_kv_cache()
            
    def forward(self, input, use_cache: bool = False, current_token_index: Optional[int] = None):
        x = self.embedding(input)

        if use_cache and current_token_index is not None:
            x = x[:, current_token_index:current_token_index + 1, :]

        attn_mask: Optional[torch.Tensor] = self.buffered_future_mask(x)
        key_padding_mask: Optional[torch.Tensor] = None

        if use_cache and current_token_index is not None:
        # when we pass kvcache, we are passing single token as input which need to attend to all previous tokens, but not the tokens after it, the size of the mask should be equal to max seq len
            key_padding_mask = look_ahead_mask(self.max_seq_len)[current_token_index:current_token_index + 1, :].to(x.device)

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
        params['dropout_rate'],
        params['batch_size']
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

