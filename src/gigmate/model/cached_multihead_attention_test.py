import random
from typing import cast
import pytest
import torch
from gigmate.model.cached_multihead_attention import CachedMultiheadAttention, create_block_mask_cached, generate_alibi_bias, get_score_mod
from gigmate.utils.constants import get_random_seed
from gigmate.utils.device import get_device

BATCH_SIZE = 2
SEQ_LEN = 10
EMBEDDING_DIM = 8
NUM_HEADS = 2
SLIDING_WINDOW_SIZE = 32

SEED = get_random_seed()
random.seed(SEED)
torch.manual_seed(SEED)


def get_attention(*args, **kwargs) -> CachedMultiheadAttention:
    torch.manual_seed(SEED)

    params = {
        "embed_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "sliding_window_size": SLIDING_WINDOW_SIZE,
    }
    params.update(kwargs)
    attention = CachedMultiheadAttention(*args, **params)

    with torch.no_grad():
        for param in attention.parameters():
            torch.nn.init.normal_(param, mean=0.2, std=0.5)
            
    return attention


# Function to generate a query vector for testing
def generate_query_vector(batch_size, seq_len, embedding_dim):
    torch.manual_seed(SEED)
    return torch.randn(batch_size, seq_len, embedding_dim)


# TODO: fix and re-enable
def skip_test_multihead_attention_shape():

    query = generate_query_vector(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
    multihead_attention = get_attention()
    attn_output, _ = multihead_attention(query)
    assert attn_output.shape == query.shape, f"Expected shape {query.shape}, got {attn_output.shape}"


def test_multihead_attention_invalid_embed_dim():
    with pytest.raises(ValueError):
        return get_attention(embed_dim=0, num_heads=2)


def test_multihead_attention_invalid_num_heads():
    with pytest.raises(ValueError):
        get_attention(embed_dim=8, num_heads=0)


def test_multihead_attention_divisibility_error():
    with pytest.raises(AssertionError):
        get_attention(embed_dim=7, num_heads=2)


def test_multihead_attention_different_input_device():
    query = generate_query_vector(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM).to("cpu")
    with pytest.raises(AssertionError):
        multihead_attention = get_attention().to("xpu")
        multihead_attention(query)


# TODO: fix and re-enable
def skip_test_multihead_attention_dtype_mismatch():
    query = generate_query_vector(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
    with pytest.raises(RuntimeError):
        multihead_attention = get_attention().to(torch.float64)
        multihead_attention(query)


# TODO: fix and re-enable
def skip_test_multihead_attention_eval():
    query = generate_query_vector(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
    multihead_attention = get_attention()
    multihead_attention.eval()
    with torch.no_grad():
        multihead_attention(query)


# TODO: fix and re-enable
def skip_test_multihead_attention_forward_pass():
    # Predefined weights and biases for reproducibility
    emb__dim = 1
    seq_len = 3
    num_heads = 1
    batch_size = 1
    query = generate_query_vector(batch_size, seq_len, emb__dim)
    multihead_attention = get_attention(embed_dim=emb__dim, num_heads=num_heads)
    
    attn_output, _ = multihead_attention(query)

    # Precomputed expected output (this should be determined ahead of time)
    expected_output = torch.tensor([[
        [-0.0090],
        [-0.0090],
        [-0.0090]
    ]])
    
    assert torch.allclose(attn_output, expected_output, atol=1e-4), "Forward pass output does not match expected value"


# TODO: fix and re-enable
def skip_test_multihead_attention_forward_pass_with_mask():
    # Predefined weights and biases for reproducibility
    embed_dim = 1
    seq_len = 3
    num_heads = 1
    batch_size = 1
    query = generate_query_vector(batch_size, seq_len, embed_dim)
    multihead_attention = get_attention(embed_dim=embed_dim, num_heads=num_heads)
    
    attn_output, _ = multihead_attention(query, sequence_lengths=[SEQ_LEN])

    # Precomputed expected output (this should be determined ahead of time)
    expected_output = torch.tensor([[
        [-0.0130],
        [-0.0090],
        [-0.0090],
    ]])
    
    assert torch.allclose(attn_output, expected_output, atol=1e-4), "Forward pass output does not match expected value"


# TODO: fix and re-enable
def skip_test_multihead_attention_forward_pass_batch():
    # Predefined weights and biases for reproducibility
    embed_dim = 1
    seq_len = 3
    num_heads = 1
    batch_size = 2
    query = generate_query_vector(batch_size, seq_len, embed_dim)
    multihead_attention = get_attention(embed_dim=embed_dim, num_heads=num_heads)

    attn_output, _ = multihead_attention(query)
    # Precomputed expected output (this should be determined ahead of time)
    expected_output = torch.tensor(
        [
            [
                [-0.0130],
                [-0.0090],
                [-0.0090]
            ],
            [
                [-0.0089],
                [0.0192],
                [0.0141]
            ]
        ]
    )

    assert torch.allclose(attn_output, expected_output, atol=1e-4), "Forward pass output does not match expected value"


# TODO: fix and re-enable
def skip_test_multihead_attention_with_cache_forward_pass():
    # Predefined weights and biases for reproducibility
    embed_dim = 1
    seq_len = 3
    num_heads = 1
    batch_size = 1

    query = generate_query_vector(batch_size, seq_len, embed_dim)
    multihead_attention_one_shot = get_attention(embed_dim=embed_dim, num_heads=num_heads)
    multihead_attention_one_shot.eval()
    multihead_attention_incremental = get_attention(embed_dim=embed_dim, num_heads=num_heads)
    multihead_attention_incremental.eval()
    
    attn_output_incremental_1, _ = multihead_attention_incremental(query, use_cache=True, cache_index=None)
    next_token = attn_output_incremental_1[:, 2:3, :]
    attn_output_incremental_2, _ = multihead_attention_incremental(next_token, use_cache=True, cache_index=3)
    
    full_seq = torch.cat([query, next_token], dim=1)
    attn_output_one_shot, _ = multihead_attention_one_shot(full_seq)
    attn_output_incremental = torch.cat([attn_output_incremental_1, attn_output_incremental_2], dim=1)
    
    assert torch.allclose(attn_output_incremental, attn_output_one_shot, atol=1e-4), "Forward pass output does not match expected value"


# TODO: fix and re-enable
def skip_test_multihead_attention_with_cache_full_forward_pass():
    embed_dim = 1
    seq_len = 3
    sliding_window_size = 3
    num_heads = 1
    batch_size = 1

    query = generate_query_vector(batch_size, seq_len, embed_dim)
    multihead_attention_one_shot = get_attention(embed_dim=embed_dim, num_heads=num_heads, sliding_window_size=sliding_window_size)
    multihead_attention_one_shot.training = False
    multihead_attention_incremental = get_attention(embed_dim=embed_dim, num_heads=num_heads, sliding_window_size=sliding_window_size)
    multihead_attention_incremental.training = False
    
    attn_output_incremental_1, _ = multihead_attention_incremental(query, use_cache=True, cache_index=None)
    next_token = attn_output_incremental_1[:, 2:3, :]
    attn_output_incremental_2, _ = multihead_attention_incremental(next_token, use_cache=True, cache_index=2)
    
    last_part_seq = torch.cat([query, next_token], dim=1)[:, 1:, :]
    attn_output_one_shot, _ = multihead_attention_one_shot(last_part_seq)
    
    assert torch.allclose(attn_output_incremental_2, attn_output_one_shot[:, -1:, :], atol=1e-4), "Forward pass output does not match expected value"


class Mod(torch.nn.Module):
    cross_attention: bool

    def __init__(self, cross_attention=True):
        super().__init__()
        self.attn = CachedMultiheadAttention(256, 4, 64, is_cross_attention=cross_attention)
        self.cross_attention = cross_attention
    
    def forward(self, x, cross_attention_input, sequence_lengths, cross_attention_sequence_lengths):
        return self.attn(
            x,
            cross_attention_input,
            cross_attention_input,
            use_cache=False,
            sequence_lengths=sequence_lengths,
            kv_sequence_lengths=cross_attention_sequence_lengths,
            inverted_key_values=self.cross_attention,
        )


def test_model_batches_are_independent():

    device = get_device()
    batch_size = 6

    if device == 'cuda' or device == 'mps':
        
        for i in range(batch_size):
            x = torch.randn((batch_size, 128, 256)).to(device)
            cross_attention_input = torch.randn((batch_size, 128, 256)).to(device)

            x.requires_grad_(True)
            cross_attention_input.requires_grad_(True)

            sequence_lengths = [80, 48, 128, 128, 128, 128, 2, 128]
            cross_attention_sequence_lengths = [90, 30, 128, 128, 3, 128]

            model = Mod().eval()

            for param in model.parameters():
                torch.nn.init.normal_(param, mean=0.0, std=0.3)
            model = cast(Mod, model).to(device)
            result = model(x, cross_attention_input, sequence_lengths, cross_attention_sequence_lengths)[0]

            batch = i
            result[batch].sum().backward()

            gradient = cast(torch.Tensor, x.grad)
            gradient_cross_attention = cast(torch.Tensor, cross_attention_input.grad)

            for idx in range(batch_size):
                if idx != batch:
                    assert gradient[idx].sum() == 0, f'The gradient for the item {idx} on which the backward pass is not computed is different than zero'
                    assert gradient_cross_attention[idx].sum() == 0, f'The cross attention gradient for the item {idx} on which the backward pass is not computed is different than zero'
                else:
                    assert gradient[idx].sum() != 0, f'The gradient for the item {idx} on which the backward pass is computed is equal to zero'
                    assert gradient_cross_attention[idx].sum() != 0, f'The cross attention gradient for the item {idx} on which the backward pass is computed is equal to zero'


def test_model_causality_is_respected():

    device = get_device()
    batch_size = 6
    max_seq_length = 128

    if device == 'cuda' or device == 'mps':
        
        for i in range(max_seq_length):
            x = torch.randn((batch_size, max_seq_length, 256)).to(device)
            x.requires_grad_(True)

            sequence_lengths = [80, 48, 128, 128, 128, 128, 2, 128]

            model = Mod(cross_attention=False).eval()

            for param in model.parameters():
                torch.nn.init.normal_(param, mean=0.0, std=0.3)

            model = cast(Mod, model).to(device)
            result = model(x, x, sequence_lengths, sequence_lengths)[0]

            token_index = i

            result[:, token_index, :].sum().backward()

            gradient = cast(torch.Tensor, x.grad)

            for token in range(max_seq_length):
                if token > token_index:
                    assert gradient[:, token].sum() == 0, f'The gradient for the token {token} that is after token {token_index} is different than zero'


def test_compilation():

    device = get_device()

    if device == 'cuda' or device == 'mps':
        x = torch.randn((2, 128, 256)).to(device)
        cross_attention_input = torch.randn((2, 128, 256)).to(device)
        sequence_lengths = [80, 48]
        cross_attention_sequence_lengths = [90, 30]

        model = Mod().eval()

        with torch.no_grad():
            for param in model.parameters():
                torch.nn.init.normal_(param, mean=0.0, std=0.3)

        model = torch.compile(model, backend='aot_eager', fullgraph=True)
        model = cast(Mod, model).to(device)
        result = model(x, cross_attention_input, sequence_lengths, cross_attention_sequence_lengths)

        result[0].sum().backward()

    else:
        print(f"Skipping test because flex attention does not support {device}")


def test_generate_alibi_bias():

    size = 4
    bias_fn = generate_alibi_bias(NUM_HEADS, size)
    tensor = torch.empty((size, size))

    for i in range(0, size):
        for j in range(0, size):
            tensor[i, j] = bias_fn(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(i), torch.tensor(j))

    assert torch.allclose(tensor, torch.tensor([
        [0.0000, -0.0625, -0.1250, -0.1875],
        [0.0625, 0.0000, -0.0625, -0.1250],
        [0.1250, 0.0625, 0.0000, -0.0625],
        [0.1875, 0.1250, 0.0625, 0.0000],
    ]), atol=1e-04)


def test_generate_alibi_bias_heads():

    size = 4
    bias_fn = generate_alibi_bias(NUM_HEADS, size)
    tensor = torch.empty((size, size))
    head_num = 2

    for i in range(0, size):
        for j in range(0, size):
            tensor[i, j] = bias_fn(torch.tensor(0), torch.tensor(0), torch.tensor(head_num), torch.tensor(i), torch.tensor(j))

    assert torch.allclose(tensor, torch.tensor([
        [0.0000, -0.0002, -0.0005, -0.0007],
        [0.0002, 0.0000, -0.0002, -0.0005],
        [0.0005, 0.0002, 0.0000, -0.0002],
        [0.0007, 0.0005, 0.0002, 0.0000],
    ]), atol=1e-04)


def test_generate_alibi_bias_inverted():

    size = 4
    bias_fn = generate_alibi_bias(NUM_HEADS, size, invert=True)
    tensor = torch.empty((size, size))

    for i in range(0, size):
        for j in range(0, size):
            tensor[i, j] = bias_fn(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(i), torch.tensor(j))
    
    assert torch.allclose(tensor, torch.tensor([
        [0.2500, 0.1875, 0.1250, 0.0625],
        [0.3125, 0.2500, 0.1875, 0.1250],
        [0.3750, 0.3125, 0.2500, 0.1875],
        [0.4375, 0.3750, 0.3125, 0.2500],
    ]), atol=1e-04)


# TODO: investigate why the test fails on CI
@pytest.mark.skip(reason="Not working on CI")
def test_causal_sliding_window_mask():
    size = 5
    window_size = 2

    mask = create_block_mask_cached(
        sliding_window_size=window_size,
        q_len=size,
        kv_len=size,
        cache_index=None,
        device='cpu',
        block_size=1,
    )

    assert torch.equal(mask.to_dense(), torch.tensor([[[
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ]]]))


def test_generate_bias():

    size = 5
    sliding_window_size = 2
    bias_fn = generate_alibi_bias(NUM_HEADS, size)
    score_mod = get_score_mod(
        causal=True,
        sliding_window_size=sliding_window_size,
        cache_index=None,
        use_cache=False,
        alibi_score_mod=bias_fn,
        sequence_lengths=[size],
        kv_sequence_lengths=[size],
        inverted_query=False,
        inverted_key_values=False,
        tgt_len=size,
        src_len=size,
        device_type='cpu',
    )
    tensor = torch.empty((size, size))

    for i in range(0, size):
        for j in range(0, size):
            tensor[i, j] = score_mod(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(i), torch.tensor(j))

    assert torch.allclose(tensor, torch.tensor([
        [0.0000, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
        [0.0625, 0.0000, float('-inf'), float('-inf'), float('-inf')],
        [float('-inf'), 0.0625, 0.0000, float('-inf'), float('-inf')],
        [float('-inf'), float('-inf'), 0.0625, 0.0000, float('-inf')],
        [float('-inf'), float('-inf'), float('-inf'), 0.0625, 0.0000],
    ]), atol=1e-04)


def test_generate_bias_with_sequence_lengths():

    size = 5
    sliding_window_size = 2
    bias_fn = generate_alibi_bias(NUM_HEADS, size)
    score_mod = get_score_mod(
        causal=True,
        sliding_window_size=sliding_window_size,
        cache_index=None,
        use_cache=False,
        alibi_score_mod=bias_fn,
        sequence_lengths=[2],
        kv_sequence_lengths=[2],
        inverted_query=False,
        inverted_key_values=False,
        tgt_len=size,
        src_len=size,
        device_type='cpu',
    )
    tensor = torch.empty((size, size))

    for i in range(0, size):
        for j in range(0, size):
            tensor[i, j] = score_mod(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(i), torch.tensor(j))

    assert torch.allclose(tensor, torch.tensor([
        [0.0000, float('-inf'), float('-inf'), float('-inf'), float('-inf')],
        [0.0625, 0.0000, float('-inf'), float('-inf'), float('-inf')],
        [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
        [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
        [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
    ]), atol=1e-04)


def test_generate_bias_with_different_sequence_lengths():

    size = 5
    sliding_window_size = 2
    bias_fn = generate_alibi_bias(NUM_HEADS, size)
    score_mod = get_score_mod(
        causal=False,
        sliding_window_size=sliding_window_size,
        cache_index=None,
        use_cache=False,
        alibi_score_mod=bias_fn,
        sequence_lengths=[4],
        kv_sequence_lengths=[2],
        inverted_query=False,
        inverted_key_values=False,
        tgt_len=size,
        src_len=size,
        device_type='cpu',
    )
    tensor = torch.empty((size, size))

    for i in range(0, size):
        for j in range(0, size):
            tensor[i, j] = score_mod(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(i), torch.tensor(j))

    assert torch.allclose(tensor, torch.tensor([
        [0.0000, -0.0625, float('-inf'), float('-inf'), float('-inf')],
        [0.0625, 0.0000, float('-inf'), float('-inf'), float('-inf')],
        [0.1250, 0.0625, float('-inf'), float('-inf'), float('-inf')],
        [0.1875, 0.1250, float('-inf'), float('-inf'), float('-inf')],
        [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
    ]), atol=1e-04)


def test_generate_bias_with_sequence_lengths_and_multiple_batches():

    size = 5
    sliding_window_size = 2
    bias_fn = generate_alibi_bias(NUM_HEADS, size)
    score_mod = get_score_mod(
        causal=False,
        sliding_window_size=sliding_window_size,
        cache_index=None,
        use_cache=False,
        alibi_score_mod=bias_fn,
        sequence_lengths=[2, 4],
        kv_sequence_lengths=[2, 4],
        inverted_query=False,
        inverted_key_values=False,
        tgt_len=size,
        src_len=size,
        device_type='cpu',
    )
    tensor = torch.empty((size, size))

    for i in range(0, size):
        for j in range(0, size):
            tensor[i, j] = score_mod(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(i), torch.tensor(j))

    assert torch.allclose(tensor, torch.tensor([
        [0.0000, -0.0625, float('-inf'), float('-inf'), float('-inf')],
        [0.0625, 0.0000, float('-inf'), float('-inf'), float('-inf')],
        [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
        [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
        [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
    ]), atol=1e-04)

    tensor = torch.empty((size, size))

    for i in range(0, size):
        for j in range(0, size):
            tensor[i, j] = score_mod(torch.tensor(0), torch.tensor(1), torch.tensor(0), torch.tensor(i), torch.tensor(j))

    assert torch.allclose(tensor, torch.tensor([
        [0.0000, -0.0625, -0.1250, -0.1875, float('-inf')],
        [0.0625, 0.0000, -0.0625, -0.1250, float('-inf')],
        [0.1250, 0.0625, 0.0000, -0.0625, float('-inf')],
        [0.1875, 0.1250, 0.0625, 0.0000, float('-inf')],
        [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
    ]), atol=1e-04)


def test_generate_bias_non_causal():

    size = 5
    sliding_window_size = 2
    bias_fn = generate_alibi_bias(NUM_HEADS, size)
    score_mod = get_score_mod(
        causal=False,
        sliding_window_size=sliding_window_size,
        cache_index=None,
        use_cache=False,
        alibi_score_mod=bias_fn,
        sequence_lengths=[size],
        kv_sequence_lengths=[size],
        inverted_query=False,
        inverted_key_values=False,
        tgt_len=size,
        src_len=size,
        device_type='cpu',
    )
    tensor = torch.empty((size, size))

    for i in range(0, size):
        for j in range(0, size):
            tensor[i, j] = score_mod(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(i), torch.tensor(j))

    assert torch.allclose(tensor, torch.tensor([
        [0.0000, -0.0625, -0.1250, -0.1875, -0.2500],
        [0.0625, 0.0000, -0.0625, -0.1250, -0.1875],
        [0.1250, 0.0625, 0.0000, -0.0625, -0.1250],
        [0.1875, 0.1250, 0.0625, 0.0000, -0.0625],
        [0.2500, 0.1875, 0.1250, 0.0625, 0.0000],
    ]), atol=1e-04)


def test_generate_bias_cross_attention():

    size = 5
    sliding_window_size = 2
    bias_fn = generate_alibi_bias(NUM_HEADS, size, invert=True)
    score_mod = get_score_mod(
        causal=False,
        sliding_window_size=sliding_window_size,
        cache_index=None,
        use_cache=False,
        alibi_score_mod=bias_fn,
        sequence_lengths=[5],
        kv_sequence_lengths=[3],
        inverted_query=False,
        inverted_key_values=True,
        tgt_len=size,
        src_len=size,
        device_type='cpu',
    )
    
    tensor = torch.empty((size, size))

    for i in range(0, size):
        for j in range(0, size):
            tensor[i, j] = score_mod(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(i), torch.tensor(j))

    assert torch.allclose(tensor, torch.tensor([
        [float('-inf'), float('-inf'), 0.1875, 0.1250, 0.0625],
        [float('-inf'), float('-inf'), 0.2500, 0.1875, 0.1250],
        [float('-inf'), float('-inf'), 0.3125, 0.2500, 0.1875],
        [float('-inf'), float('-inf'), 0.3750, 0.3125, 0.2500],
        [float('-inf'), float('-inf'), 0.4375, 0.3750, 0.3125],
    ]), atol=1e-04)


def test_generate_bias_encoder_self_attention():

    size = 5
    sliding_window_size = 3
    bias_fn = generate_alibi_bias(NUM_HEADS, size, invert=False)
    score_mod = get_score_mod(
        causal=True,
        sliding_window_size=sliding_window_size,
        cache_index=None,
        use_cache=False,
        alibi_score_mod=bias_fn,
        sequence_lengths=[size - 1],
        kv_sequence_lengths=[size - 1],
        inverted_query=True,
        inverted_key_values=True,
        tgt_len=size,
        src_len=size,
        device_type='cpu',
    )
    tensor = torch.empty((size, size))

    for i in range(0, size):
        for j in range(0, size):
            tensor[i, j] = score_mod(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(i), torch.tensor(j))

    assert torch.allclose(tensor, torch.tensor([
        [float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
        [float('-inf'), 0.0000, float('-inf'), float('-inf'), float('-inf')],
        [float('-inf'), 0.0625, 0.0000, float('-inf'), float('-inf')],
        [float('-inf'), 0.1250, 0.0625, 0.0000, float('-inf')],
        [float('-inf'), float('-inf'), 0.1250, 0.0625, 0.0000],
    ]), atol=1e-04)