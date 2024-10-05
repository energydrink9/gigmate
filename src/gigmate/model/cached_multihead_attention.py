
from functools import lru_cache
import math
from typing import Callable, List, Optional, Tuple, cast
import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.overrides
import torch.utils.backend_registration
import torch.utils._python_dispatch
import torch._C
import torch.fx.experimental.proxy_tensor
#from torch.nn.attention.flex_attention import flex_attention, and_masks, create_block_mask, _mask_mod_signature, _score_mod_signature, BlockMask
import xformers.ops
import xformers.ops.fmha.attn_bias
import xformers.components.attention.attention_patterns

pad = torch._C._nn.pad,
linear = torch._C._nn.linear

def generate_alibi_bias(H: int):# -> _score_mod_signature:
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


def generate_sliding_window_mask_mod(window_size: int, cache_index: int):# -> _mask_mod_signature:
    """Generates a sliding window attention mask with a given window size.
    Args:
        window_size: The size of the sliding window.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking.
    """

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    
    def sliding_window(b, h, q_idx, kv_idx):
        return q_idx - kv_idx <= window_size
    
    def incremental_sliding_window(b, h, q_idx, kv_idx):
        return cache_index - kv_idx <= window_size

    def incremental_padding_mask(b, h, q_idx, kv_idx):
        return cache_index >= kv_idx

    if cache_index is not None:
        return None#and_masks(incremental_sliding_window, incremental_padding_mask)
    else:
        return None#and_masks(sliding_window, causal_mask)

@lru_cache(maxsize=1)
def create_block_mask_cached(sliding_window_size: int, q_len: int, kv_len: int, cache_index: int, device: str):# -> BlockMask:
    return None
    # return create_block_mask(
    #     mask_mod=generate_sliding_window_mask_mod(sliding_window_size, cache_index),
    #     B=None,
    #     H=None,
    #     Q_LEN=q_len,
    #     KV_LEN=kv_len,
    #     device=device,
    # )

@lru_cache(maxsize=1)
def get_slopes(n: int):
    def get_slopes_power_of_2(n: int) -> List[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    # In the paper, we only train models that have 2^a heads for some a. This function has
    # some good properties that only occur when the input is a power of 2. To maintain that even
    # when the number of heads is not a power of 2, we use this workaround.
    if math.log2(n) % 1 == 0:
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

def get_alibi_bias(mask_shape) -> Tensor:
    maxpos = mask_shape[1]
    attn_heads = mask_shape[0]
    slopes = torch.Tensor(get_slopes(attn_heads))

    # In the next line, the part after the * is what constructs the diagonal matrix
    # (right matrix in Figure 3 in the paper).
    # If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3,
    # but one where all rows are identical.
    # This works because the softmax operation is invariant to translation,
    # and our bias functions are always linear.
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(
        0
    ).unsqueeze(0).expand(attn_heads, -1, -1)
    alibi = alibi.view(attn_heads, 1, maxpos)
    return alibi

def create_mask(shape: Tuple[int, int], predicate: Callable[[Tensor, Tensor], Tensor]):
    # Create 2D index grids for query and key-value indices
    q_idx, kv_idx = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij')
    
    # Apply the predicate to generate the mask
    mask = predicate(q_idx, kv_idx).to(torch.bool)
    
    # Convert the boolean mask: True to 0, False to -inf
    return torch.where(mask, torch.tensor(0.0), torch.tensor(float('-inf')))

def get_disabled_cache_attn_mask(size: tuple[int, int], sliding_window_size: int, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    causal_mask = create_mask(size, lambda q_idx, kv_idx: q_idx >= kv_idx)
    sliding_window_mask = create_mask(size, lambda q_idx, kv_idx: q_idx - kv_idx <= sliding_window_size)

    if key_padding_mask is None:
        return causal_mask + sliding_window_mask
    else:
        return causal_mask + sliding_window_mask + key_padding_mask

def get_cache_attn_mask(size: tuple[int, int], cache_index: int) -> torch.Tensor:
    causal_mask = create_mask(size, lambda _, kv_idx: kv_idx <= cache_index)

    return causal_mask

class LocalAttentionFromBottomRightMaskWithBiasTensor(xformers.ops.fmha.attn_bias.LocalAttentionFromBottomRightMask, xformers.ops.fmha.attn_bias.LowerTriangularMaskWithTensorBias):
    pass

# def get_xformers_attn_bias(sliding_window_size: int, bias_tensor: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
#     if key_padding_mask is None:
#         bias = LocalAttentionFromBottomRightMaskWithBiasTensor(
#             window_left=sliding_window_size,
#             window_right=0,
#         )
#         bias._subtensor=bias_tensor,
#     else:
#         return xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
#             q_seqlen=key_padding_mask,
#         ).make_causal().make_local_attention(window_size=sliding_window_size)

# def get_cache_xformers_attn_bias(size, cache_index: int):
#     causal_mask = create_mask(size, lambda _, kv_idx: kv_idx <= cache_index)

#     return causal_mask

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

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)
        self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=False)
        self._reset_parameters()

        # ALiBi slope
        m = torch.arange(1, self.num_heads + 1)
        m = 1.0 / torch.pow(2, m / self.num_heads)
        self.register_buffer("m", m.unsqueeze(0).unsqueeze(1).unsqueeze(1))  # [1, 1, 1, num_heads]

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

    def forward(
            self,
            query: Tensor,
            use_cache: bool = False,
            cache: Optional[Tensor] = None,
            cache_index: Optional[int] = None,
            key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:

        attn_output, updated_kv_cache = self.multi_head_attention_forward(
            query, self.embed_dim, self.num_heads,
            in_proj_weight=self.in_proj_weight,
            out_proj_weight=self.out_proj.weight,
            sliding_window_size=self.sliding_window_size,
            use_cache=use_cache,
            cache=None if self.training or not use_cache else cache,
            cache_index=cache_index,
            key_padding_mask=key_padding_mask,
        )

        return attn_output.transpose(1, 0), updated_kv_cache
        
    def multi_head_attention_forward(
        self,
        query: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Optional[Tensor],
        out_proj_weight: Tensor,
        sliding_window_size: int,
        use_cache: bool = False,
        cache: Optional[Tensor] = None,
        cache_index: Optional[int] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        query = key = value = query.transpose(1, 0)

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
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, in_proj_weight, None)

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

        mask_size = (tgt_len, src_len)
        alibi_bias = get_alibi_bias((num_heads, src_len, tgt_len)).to(q.device)

        # if q.device == 'cuda':
            # TODO: Try out flex attention and compare performance
            # block_mask = create_block_mask_cached(sliding_window_size, tgt_len, src_len, cache_index, device=q.device.type)
            # attn_output = cast(Tensor, flex_attention(q, k, v, block_mask=block_mask, score_mod=self.alibi_score_mod))
            # attn_bias = get_cache_xformers_attn_bias(mask_size, cache_index) if cache_index is not None else get_xformers_attn_bias(sliding_window_size, alibi_bias)
            # attn_bias = attn_bias.to(q.device)
            # attn_output = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)

        # else:
        attn_bias = get_cache_attn_mask(mask_size, cache_index) if cache_index is not None else get_disabled_cache_attn_mask(mask_size, sliding_window_size, key_padding_mask)
        attn_bias = attn_bias.to(q.device)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias + alibi_bias)
        
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
    w: Tensor,
    b: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    E = q.size(-1)
    # self-attention
    proj = linear(q, w, b)
    # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
    proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
    return proj[0], proj[1], proj[2]
