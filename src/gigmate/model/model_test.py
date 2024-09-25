import random
import pytest
import torch
from gigmate.model.model import TransformerModel
from gigmate.utils.constants import get_random_seed

NUM_LAYERS = 2
BATCH_SIZE = 2
SEQ_LEN = 10
EMBEDDING_DIM = 8
NUM_HEADS = 2
MAX_SEQ_LEN = 32
VOCAB_SIZE = 3

SEED = get_random_seed()
random.seed(SEED)
torch.manual_seed(SEED)

def get_token(logits: torch.Tensor):
    return torch.argmax(logits, dim=-1, keepdim=True)

# Function to generate a query vector for testing
def generate_query_vector(batch_size, seq_len):
    torch.manual_seed(SEED)
    val = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    return val

def get_model(*args, **kwargs):
    torch.manual_seed(SEED)
    params = {
        "num_layers": NUM_LAYERS,
        "d_model": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "dff": EMBEDDING_DIM * 4,
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN
    }
    params.update(kwargs)

    model = TransformerModel(*args, **params)
    model.eval()

    with torch.no_grad():
        for param in model.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.3)

    return model

def test_model_shape():

    query = generate_query_vector(BATCH_SIZE, SEQ_LEN)
    model = get_model()
    output = model(query)
    assert output.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), f"Expected shape {(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)}, got {output.shape}"

def test_model_invalid_embed_dim():
    with pytest.raises(ValueError):
        get_model(d_model=0)

def test_model_invalid_num_heads():
    with pytest.raises(ValueError):
        get_model(num_heads=0)

def test_model_different_input_device():
    query = generate_query_vector(BATCH_SIZE, SEQ_LEN).to("cpu")
    with pytest.raises(AssertionError):
        model = get_model().to("cuda")
        model(query)

def test_model_dtype_mismatch():
    query = generate_query_vector(BATCH_SIZE, SEQ_LEN).to(torch.float64)
    with pytest.raises(RuntimeError):
        model = get_model()
        model(query)

def test_model_eval():
    query = generate_query_vector(BATCH_SIZE, SEQ_LEN)
    model = get_model()
    model.eval()
    with torch.no_grad():
        model(query)

def test_model_forward_pass():

    emb__dim = 2
    seq_len = 3
    num_heads = 1
    batch_size = 1
    query = generate_query_vector(batch_size, seq_len)
    model = get_model(d_model=emb__dim, num_heads=num_heads, max_seq_len=MAX_SEQ_LEN)
    
    output = get_token(model(query))

    # Precomputed expected output (this should be determined ahead of time)
    expected_output = torch.tensor([[
         [0],
         [1],
         [1]
    ]])
    
    assert torch.allclose(output, expected_output, atol=1e-4), "Forward pass output does not match expected value"

def test_scripted_model_forward_pass():

    emb__dim = 2
    seq_len = 3
    num_heads = 1
    batch_size = 1
    query = generate_query_vector(batch_size, seq_len)
    model = get_model(d_model=emb__dim, num_heads=num_heads, max_seq_len=MAX_SEQ_LEN)
    model = torch.jit.script(model)
    
    output = get_token(model(query))

    # Precomputed expected output (this should be determined ahead of time)
    expected_output = torch.tensor([[
         [0],
         [1],
         [1]
    ]])
    
    assert torch.allclose(output, expected_output, atol=1e-4), "Forward pass output does not match expected value"

def test_model_forward_pass_batch():
    # Predefined weights and biases for reproducibility
    emb__dim = 2
    seq_len = 3
    num_heads = 1
    batch_size = 2
    query = generate_query_vector(batch_size, seq_len)
    model = get_model(d_model=emb__dim, num_heads=num_heads)
    
    output = get_token(model(query))

    expected_output = torch.tensor(
        [[
            [0],
            [1],
            [1]
        ],
        [
            [0],
            [1],
            [1]
        ]]
    )

    assert torch.allclose(output, expected_output, atol=1e-4), "Forward pass output does not match expected value"

def test_model_with_cache_forward_pass():
    # Predefined weights and biases for reproducibility
    emb__dim = 4
    seq_len = 3
    num_heads = 2
    batch_size = 1
    num_layers = 4

    query = generate_query_vector(batch_size, seq_len)
    model_one_shot = get_model(d_model=emb__dim, num_heads=num_heads, num_layers=num_layers)
    model_one_shot.training = False
    model_incremental = get_model(d_model=emb__dim, num_heads=num_heads, num_layers=num_layers)
    model_incremental.training = False
    print('inc 1')
    output_incremental_1 = model_incremental(query, use_cache=True, current_token_index=None)
    next_token = get_token(output_incremental_1[:, -1, :])
    full_seq = torch.cat([query, next_token], dim=1)
    print('-------------')
    print('inc 2')
    output_incremental_2 = model_incremental(full_seq, use_cache=True, current_token_index=3)
    print('-------------')
    print('one shot')
    output_one_shot = model_one_shot(full_seq)

    output_incremental = torch.cat([output_incremental_1, output_incremental_2], dim=1)
    assert torch.allclose(output_incremental, output_one_shot, atol=1e-6), "Forward pass output does not match expected value"
