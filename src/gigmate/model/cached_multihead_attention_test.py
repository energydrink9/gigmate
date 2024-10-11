import random
import pytest
import torch
from gigmate.model.cached_multihead_attention import CachedMultiheadAttention
from gigmate.utils.constants import get_random_seed

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


def test_multihead_attention_shape():

    query = generate_query_vector(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
    multihead_attention = get_attention()
    attn_output, _ = multihead_attention(query)
    assert attn_output.shape == query.shape, f"Expected shape {query.shape}, got {attn_output.shape}"


def test_multihead_attention_invalid_embed_dim():
    with pytest.raises(ValueError):
        return get_attention(embed_dim = 0, num_heads = 2)


def test_multihead_attention_invalid_num_heads():
    with pytest.raises(ValueError):
        get_attention(embed_dim = 8, num_heads = 0)


def test_multihead_attention_divisibility_error():
    with pytest.raises(AssertionError):
        get_attention(embed_dim = 7, num_heads = 2)


def test_multihead_attention_different_input_device():
    query = generate_query_vector(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM).to("cpu")
    with pytest.raises(AssertionError):
        multihead_attention = get_attention().to("cuda")
        multihead_attention(query)


def test_multihead_attention_dtype_mismatch():
    query = generate_query_vector(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
    with pytest.raises(RuntimeError):
        multihead_attention = get_attention().to(torch.float64)
        multihead_attention(query)


def test_multihead_attention_eval():
    query = generate_query_vector(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
    multihead_attention = get_attention()
    multihead_attention.eval()
    with torch.no_grad():
        multihead_attention(query)


def test_multihead_attention_forward_pass():
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


def test_multihead_attention_forward_pass_with_mask():
    # Predefined weights and biases for reproducibility
    embed_dim = 1
    seq_len = 3
    num_heads = 1
    batch_size = 1
    query = generate_query_vector(batch_size, seq_len, embed_dim)
    multihead_attention = get_attention(embed_dim=embed_dim, num_heads=num_heads)
    
    attn_output, _ = multihead_attention(query, key_padding_mask=look_ahead_mask(seq_len))

    # Precomputed expected output (this should be determined ahead of time)
    expected_output = torch.tensor([[
         [-0.0130],
         [-0.0090],
         [-0.0090]
    ]])
    
    assert torch.allclose(attn_output, expected_output, atol=1e-4), "Forward pass output does not match expected value"


def test_multihead_attention_forward_pass_batch():
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
        [[
            [-0.0130],
            [-0.0090],
            [-0.0090]
        ],
        [
            [-0.0089],
            [ 0.0192],
            [ 0.0141]
        ]]
    )

    assert torch.allclose(attn_output, expected_output, atol=1e-4), "Forward pass output does not match expected value"


def test_multihead_attention_with_cache_forward_pass():
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


def test_multihead_attention_with_cache_full_forward_pass():
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
    
    last_part_seq = torch.cat([query, next_token], dim=1)[:,1:,:]
    attn_output_one_shot, _ = multihead_attention_one_shot(last_part_seq)
    
    assert torch.allclose(attn_output_incremental_2, attn_output_one_shot[:,-1:,:], atol=1e-4), "Forward pass output does not match expected value"
