import sys
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.types import Device
import torch.overrides
import torch._jit_internal as jit
import torch.utils.backend_registration
import torch.utils._python_dispatch
import torch._C
from torch.types import _dtype as DType

pad = torch._C._nn.pad,
linear = torch._C._nn.linear
scaled_dot_product_attention = torch._C._nn.scaled_dot_product_attention

def look_ahead_mask(size: int) -> torch.Tensor:  
    mask = torch.triu(torch.ones(size, size), diagonal=1) != 0
    return mask

class CachedMultiheadAttention(torch.nn.Module):

    kv_cache: Optional[Tensor]

    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)
        self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=False)
        
        self.register_buffer('kv_cache', None, True)

        self._reset_parameters()

    @torch.jit.export
    def reset_kv_cache(self):
        self.kv_cache = None

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

    def forward(
            self,
            query: Tensor,
            attn_mask: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            is_causal : bool = False,
            use_cache: bool = False,
            current_token_index: Optional[int] = None) -> Tuple[Tensor, Optional[Tensor]]:

        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))):
            why_not_fast_path = "floating-point masks are not supported for fast path."
        
        is_batched = query.dim() == 3

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        key = query
        value = query

        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = ("some Tensor argument's device is neither one of "
                                     f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
            elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
                
        any_nested = query.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        kv_cache: Optional[Tensor] = self.kv_cache

        attn_output, updated_kv_cache = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            in_proj_weight=self.in_proj_weight,
            dropout_p=self.dropout, out_proj_weight=self.out_proj.weight,
            max_seq_len=self.max_seq_len,
            training=self.training,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            use_cache=use_cache,
            kv_cache=None if self.training or not use_cache else kv_cache,
            current_token_index=current_token_index
        )

        if not self.training and use_cache:
            self.kv_cache = updated_kv_cache
        
        if is_batched:
            return attn_output.transpose(1, 0), None
        else:
            return attn_output, None


def _check_arg_device(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.device.type in ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
    return True


def _arg_requires_grad(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.requires_grad
    return False


def _is_make_fx_tracing():
    if not jit.is_scripting():
        torch_dispatch_mode_stack = torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        return any(type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode for x in torch_dispatch_mode_stack)
    else:
        return False
    

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    max_seq_len: int,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    use_cache: bool = False,
    kv_cache: Optional[Tensor] = None,
    current_token_index: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

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
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, None)

    updated_cache: Optional[Tensor] = None

    if use_cache:
        if kv_cache is None:
            # Initialize cache
            padding_value = 0.
            cache_k = torch.full((max_seq_len, bsz, embed_dim), padding_value)
            cache_v = torch.full((max_seq_len, bsz, embed_dim), padding_value)

            cache_k[:k.size(0), :, :] = k
            cache_v[:v.size(0), :, :] = v

            updated_cache = torch.stack((cache_k, cache_v), dim=0).to(q.device.type)

        else:
            if current_token_index is None:
                raise ValueError("current_token_index must be provided when using kv_cache")
    
            old_k = kv_cache[0]
            old_v = kv_cache[1]
            
            # when the current_token_index == max_seq_len - 1, it means that the cache is full
            # and we need to shift the cache to the left by 1 to make room for the new token
            if current_token_index == max_seq_len - 1:
                new_k = torch.roll(old_k, shifts=-1, dims=0)
                new_v = torch.roll(old_v, shifts=-1, dims=0)

                new_k[-1:, :, :] = k
                new_v[-1:, :, :] = v

            else:                
                # Compute the number of old cache elements to keep
                new_k = old_k
                new_v = old_v

                new_k[current_token_index:current_token_index + 1, :, :] = k
                new_v[current_token_index:current_token_index + 1, :, :] = v

            updated_cache = torch.stack((new_k, new_v), dim=0)

            k = new_k
            v = new_v

    if attn_mask is not None and not use_cache:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    # attn_mask can be either (L,S) or (N*num_heads, L, S)
    # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
    # in order to match the input for SDPA of (N, num_heads, L, S)
    if attn_mask is not None:
        if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(0)
        else:
            attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

    q = q.view(bsz, num_heads, tgt_len, head_dim)
    k = k.view(bsz, num_heads, src_len, head_dim)
    v = v.view(bsz, num_heads, src_len, head_dim)

    attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

    attn_output = linear(attn_output, out_proj_weight, None)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
    
    if not is_batched:
        # squeeze the output if input was unbatched
        attn_output = attn_output.squeeze(1)

    return attn_output, updated_cache

def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

    return is_batched

def _canonical_mask(
        mask: Optional[Tensor],
        mask_name: str,
        other_type: Optional[DType],
        other_name: str,
        target_type: DType,
        check_other: bool = True,
) -> Optional[Tensor]:

    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        
        if not _mask_is_float:
            mask = (
                torch.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float("-inf"))
            )
    return mask

def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    E = q.size(-1)
    # self-attention
    proj = linear(q, w, b)
    # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
    proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
    return proj[0], proj[1], proj[2]

def _none_or_dtype(input: Optional[Tensor]) -> Optional[DType]:
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")
