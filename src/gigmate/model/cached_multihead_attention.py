
import functools
from typing import Callable, List, Optional, Tuple, cast
import torch
from torch import Tensor
from torch.nn.modules.linear import Linear
import torch.overrides
import torch.utils.backend_registration
import torch.utils._python_dispatch
import torch._C
import torch.fx.experimental.proxy_tensor
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, _mask_mod_signature, _score_mod_signature, BlockMask

pad = torch._C._nn.pad,
linear = torch._C._nn.linear


def generate_alibi_bias(H: int) -> _score_mod_signature:
    """Returns an alibi bias score_mod given the number of heads H

    Args:
        H: number of heads

    Returns:
        alibi_bias: alibi bias score_mod
    """

    def alibi_mod(score, b, h, q_idx, kv_idx):
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = (q_idx - kv_idx) * scale
        return score + bias

    return alibi_mod


def get_alibi_mask(shape: Tuple[int, int, int, int], fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor], device: str) -> Tensor:

    indices = [torch.arange(dim, device=device) for dim in shape]    
    grid = torch.meshgrid(*indices, indexing='ij')
    result = fn(*grid)
    
    return result.to(device)


def generate_sliding_window_mask_mod(window_size: int, cache_index: Optional[int], device_type: str) -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """
    
    def causal_sliding_padded_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor):
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= window_size)

    # Converting the cache index to a tensor, because PyTorch raises an error it's not
    cache_index_tensor = torch.tensor(cache_index, device=device_type) if cache_index is not None else None
    
    def incremental_causal_sliding_mask(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor):
        return (kv_idx >= cast(Tensor, cache_index_tensor) - window_size) & (kv_idx <= cast(Tensor, cache_index_tensor))

    if cache_index is not None:
        return incremental_causal_sliding_mask
    else:
        return causal_sliding_padded_mask


@functools.lru_cache(maxsize=2)
def create_block_mask_cached(sliding_window_size: int, q_len: int, kv_len: int, cache_index: Optional[int], device: str) -> BlockMask:
    return create_block_mask(
        mask_mod=generate_sliding_window_mask_mod(sliding_window_size, cache_index, device),
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
        _compile=device != 'mps',
    )


class CachedMultiheadAttention(torch.nn.Module):

    kv_cache: Optional[Tensor]
    sliding_window_size: int
    embed_dim: int
    num_heads: int
    head_dim: int

    def __init__(self, embed_dim: int, num_heads: int, sliding_window_size: int) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sliding_window_size = sliding_window_size
        self.alibi_score_mod = generate_alibi_bias(num_heads)

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim, bias=False)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            use_cache: bool = False,
            cache: Optional[Tensor] = None,
            cache_index: Optional[int] = None,
            sequence_lengths: Optional[List[int]] = None,
            kv_sequence_lengths: Optional[List[int]] = None,
            inverted_query: bool = False,
            inverted_key_values: bool = False,
            causal: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        attn_output, updated_kv_cache = self.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            in_proj_weight=self.in_proj.weight,
            out_proj_weight=self.out_proj.weight,
            sliding_window_size=self.sliding_window_size,
            use_cache=use_cache,
            cache=None if self.training or not use_cache else cache,
            cache_index=cache_index,
            sequence_lengths=sequence_lengths,
            kv_sequence_lengths=kv_sequence_lengths,
            inverted_query=inverted_query,
            inverted_key_values=inverted_key_values,
            causal=causal,
        )

        return attn_output.transpose(1, 0), updated_kv_cache
        
    # TODO: refactor this function
    def multi_head_attention_forward(  # noqa: C901
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        out_proj_weight: Tensor,
        sliding_window_size: int,
        use_cache: bool = False,
        cache: Optional[Tensor] = None,
        cache_index: Optional[int] = None,
        sequence_lengths: Optional[List[int]] = None,
        kv_sequence_lengths: Optional[List[int]] = None,
        inverted_query: bool = False,
        inverted_key_values: bool = False,
        causal: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        query = query.transpose(1, 0)
        key = key.transpose(1, 0)
        value = value.transpose(1, 0)
        
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        assert embed_dim == embed_dim_to_check, \
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        #
        # compute in-projection
        #
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight)

        updated_cache: Optional[Tensor] = None

        if use_cache:
            updated_cache = update_kv_cache(cache, q, k, v, cache_index, sliding_window_size, bsz, embed_dim)
            if cache is not None:
                k = updated_cache[0]
                v = updated_cache[1]
            
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        # update source sequence length after adjustments
        src_len = k.size(1)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        if causal:
            block_mask = create_block_mask_cached(sliding_window_size, tgt_len, src_len, cache_index, device=q.device.type)
        else:
            block_mask = None

        if sequence_lengths is not None:
            sequence_lengths_tensor = torch.tensor(sequence_lengths, device=q.device.type)
        else:
            sequence_lengths_tensor = None

        if kv_sequence_lengths is not None:
            kv_sequence_lengths_tensor = torch.tensor(kv_sequence_lengths, device=q.device.type)
        else:
            kv_sequence_lengths_tensor = None

        tgt_len_tensor = torch.tensor(tgt_len, device=q.device.type)
        src_len_tensor = torch.tensor(src_len, device=q.device.type)

        if kv_sequence_lengths is not None:
            for length in kv_sequence_lengths:
                if length <= 0:
                    print('Warning: the length of kv sequence length is less or equal to 0')

        def score_mod(score, b, h, q_idx, kv_idx) -> Tensor:
            q_idx_override = torch.tensor(cache_index, device=q_idx.device) if use_cache and cache_index is not None else q_idx
            alibi_score = self.alibi_score_mod(score, b, h, q_idx_override, kv_idx)
            
            padding_score_q = None
            padding_score_kv = None

            if sequence_lengths_tensor is not None:
                sequence_length = sequence_lengths_tensor[b]
                if inverted_query is True:
                    padding_score_q = torch.where(q_idx >= tgt_len_tensor - sequence_length, 0., float('-inf'))
                else:
                    padding_score_q = torch.where(q_idx < sequence_length, 0., float('-inf'))

            if kv_sequence_lengths_tensor is not None:
                kv_sequence_length = kv_sequence_lengths_tensor[b]
                if inverted_key_values is True:
                    padding_score_kv = torch.where(kv_idx >= src_len_tensor - kv_sequence_length, 0., float('-inf'))
                else:
                    padding_score_kv = torch.where(kv_idx < kv_sequence_length, 0., float('-inf'))

            if padding_score_q is not None and padding_score_kv is not None:
                return alibi_score + padding_score_q + padding_score_kv
            elif padding_score_q is not None:
                return alibi_score + padding_score_q
            elif padding_score_kv is not None:
                return alibi_score + padding_score_kv
            else:
                return alibi_score

        attn_output = cast(Tensor, flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod))
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

        attn_output = linear(attn_output, out_proj_weight, None)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        return attn_output, updated_cache


def update_kv_cache(kv_cache: Optional[Tensor], q: Tensor, k: Tensor, v: Tensor, cache_index: Optional[int], cache_length: int, bsz: int, embed_dim: int) -> torch.Tensor:

    if kv_cache is None:
        # Initialize cache
        padding_value = 0.
        cache_k = torch.full((cache_length, bsz, embed_dim), padding_value)
        cache_v = torch.full((cache_length, bsz, embed_dim), padding_value)

        # Take at most cache_length elements from the end of k and v and put them at the start of the cache
        k_elements_to_modify = min(k.size(0), cache_length)
        v_elements_to_modify = min(k.size(0), cache_length)
        cache_k[:k_elements_to_modify, :, :] = k[-k_elements_to_modify:, :, :]
        cache_v[:v_elements_to_modify, :, :] = v[-v_elements_to_modify:, :, :]

        return torch.stack((cache_k, cache_v), dim=0).to(q.device.type)

    else:
        if cache_index is None:
            raise ValueError("cache_index must be provided when using kv_cache")

        old_k = kv_cache[0]
        old_v = kv_cache[1]
        
        # when the cache_index == cache_length - 1, it means that the cache is full
        # and we need to shift the cache to the left by 1 to make room for the new token
        if cache_index == cache_length - 1:
            new_k = torch.roll(old_k, shifts=-1, dims=0)
            new_v = torch.roll(old_v, shifts=-1, dims=0)

            new_k[-1:, :, :] = k
            new_v[-1:, :, :] = v

        else:                
            new_k = old_k
            new_v = old_v
            new_k[cache_index:cache_index + 1, :, :] = k
            new_v[cache_index:cache_index + 1, :, :] = v

        return torch.stack((new_k, new_v), dim=0)


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    E = q.size(-1)
    if q is k:
        # self-attention
        proj = linear(q, w, None)
        # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
        proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        return proj[0], proj[1], proj[2]

    else:
        # cross attention
        w_q, w_kv = w.split([E, E * 2])
        q_proj = linear(q, w_q, None)
        kv_proj = linear(k, w_kv, None)
        # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
        kv_proj = (
            kv_proj.unflatten(-1, (2, E))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        return (q_proj, kv_proj[0], kv_proj[1])
