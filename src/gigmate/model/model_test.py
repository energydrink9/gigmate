import random
import pytest
import torch
from gigmate.dataset.dataset import SequenceLengths
from gigmate.model.model import TransformerModel, generate_causal_sliding_mask, get_key_padding_mask, positional_encoding
from gigmate.utils.constants import get_random_seed
from gigmate.utils.device import get_device

NUM_LAYERS = 2
BATCH_SIZE = 2
SEQ_LEN = 10
EMBEDDING_DIM = 8
CODEBOOKS = 2
NUM_HEADS = 2
MAX_SEQ_LEN = 32
MAX_DECODER_SEQ_LEN = 32
VOCAB_SIZE = 3
SLIDING_WINDOW_SIZE = 32

SEED = get_random_seed()
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = get_device()


def get_token(logits: torch.Tensor):
    return torch.argmax(logits, dim=-1, keepdim=True)


# Function to generate a query vector for testing
def generate_query_vector(batch_size: int, seq_len: int, codebooks: int):
    torch.manual_seed(SEED)
    val = torch.randint(0, VOCAB_SIZE, (batch_size, codebooks, seq_len))
    return val


def get_model(*args, **kwargs):
    torch.manual_seed(SEED)
    params = {
        "encoder_layers": NUM_LAYERS,
        "decoder_layers": NUM_LAYERS,
        "d_model": EMBEDDING_DIM,
        "codebooks": CODEBOOKS,
        "num_heads": NUM_HEADS,
        "dff": EMBEDDING_DIM * 4,
        "vocab_size": VOCAB_SIZE,
        "sliding_window_size": SLIDING_WINDOW_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "max_decoder_seq_len": MAX_DECODER_SEQ_LEN,
    }
    params.update(kwargs)

    model = TransformerModel(*args, **params)

    with torch.no_grad():
        for param in model.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.3)

    return model


# TODO: fix and re-enable
def skip_test_model_shape():

    query = generate_query_vector(BATCH_SIZE, SEQ_LEN, CODEBOOKS)
    model = get_model()
    output = model(query)
    assert output.shape == (BATCH_SIZE, CODEBOOKS, SEQ_LEN, VOCAB_SIZE), f"Expected shape {(BATCH_SIZE, CODEBOOKS, SEQ_LEN, VOCAB_SIZE)}, got {output.shape}"


# TODO: fix and re-enable
def skip_test_model_invalid_embed_dim():
    with pytest.raises(ValueError):
        get_model(d_model=0)


# TODO: fix and re-enable
def skip_test_model_invalid_num_heads():
    with pytest.raises(ValueError):
        get_model(num_heads=0)


# TODO: fix and re-enable
def skip_test_model_different_input_device():
    query = generate_query_vector(BATCH_SIZE, SEQ_LEN, CODEBOOKS).to("cpu")
    with pytest.raises(AssertionError):
        model = get_model().to("cuda")
        model(query)


# TODO: fix and re-enable
def skip_test_model_dtype_mismatch():
    query = generate_query_vector(BATCH_SIZE, SEQ_LEN, CODEBOOKS).to(torch.float64)
    with pytest.raises(RuntimeError):
        model = get_model()
        model(query)


def test_model_eval():
    if DEVICE == 'cuda' or DEVICE == 'mps':

        query = generate_query_vector(BATCH_SIZE, MAX_DECODER_SEQ_LEN, CODEBOOKS).to(DEVICE)
        conditioning_query = generate_query_vector(BATCH_SIZE, MAX_SEQ_LEN, CODEBOOKS).to(DEVICE)
        sequence_lengths = SequenceLengths(full_track=[12, 2], stem=[32, 10])
        model = get_model(d_model=EMBEDDING_DIM, num_heads=NUM_HEADS).to(DEVICE)
        model.eval()
        
        model(query, conditioning_query, sequence_lengths=sequence_lengths)


def test_model_forward_pass():
    if DEVICE == 'cuda' or DEVICE == 'mps':

        query = generate_query_vector(BATCH_SIZE, MAX_DECODER_SEQ_LEN, CODEBOOKS).to(DEVICE)
        conditioning_query = generate_query_vector(BATCH_SIZE, MAX_SEQ_LEN, CODEBOOKS).to(DEVICE)
        sequence_lengths = SequenceLengths(full_track=[12, 2], stem=[32, 10])
        model = get_model(d_model=EMBEDDING_DIM, num_heads=NUM_HEADS).to(DEVICE)
        model.eval()

        get_token(model(query, conditioning_query, sequence_lengths=sequence_lengths)[0])


def test_model_pos_encodings():

    if DEVICE == 'cuda' or DEVICE == 'mps':

        pos_enc = positional_encoding(5, 4)
        
        expected = torch.tensor([[
            [0.0000, 1.0000, 0.0000, 1.0000],
            [0.8415, 0.5403, 0.0100, 0.9999],
            [0.9093, -0.4161, 0.0200, 0.9998],
            [0.1411, -0.9900, 0.0300, 0.9996],
            [-0.7568, -0.6536, 0.0400, 0.9992]
        ]])

        print(pos_enc)

        assert torch.allclose(pos_enc, expected, atol=1e-4)
        

def test_compiled_model():

    device = get_device()

    if device == 'cuda' or device == 'mps':

        emb__dim = 2
        num_heads = 1
        batch_size = 2
        query = generate_query_vector(batch_size, MAX_DECODER_SEQ_LEN, CODEBOOKS).to(device)
        conditioning_query = generate_query_vector(batch_size, MAX_SEQ_LEN, CODEBOOKS).to(device)
        sequence_lengths = SequenceLengths(full_track=[22, 32], stem=[10, 2])
        model = get_model(d_model=emb__dim, num_heads=num_heads).to(device)
        model = torch.compile(model, backend='aot_eager', fullgraph=True)
        
        result = model(query, conditioning_query, sequence_lengths=sequence_lengths)

        result[0].sum().backward()
    
    else:
        print(f"Skipping test because flex attention does not support {device}")


# TODO: fix and re-enable
def skip_test_model_batch():
    emb__dim = 2
    seq_len = 3
    num_heads = 1
    batch_size = 2
    query = generate_query_vector(batch_size, seq_len, CODEBOOKS)
    model = get_model(d_model=emb__dim, num_heads=num_heads)
    output = get_token(model(query))

    expected_output = torch.tensor(
        [
            [
                [
                    [2],
                    [2],
                    [2]
                ],
                [
                    [0],
                    [0],
                    [0]
                ]
            ],
            [
                [
                    [2],
                    [2],
                    [2]
                ],
                [
                    [0],
                    [0],
                    [0]
                ]
            ],        
        ]
    )

    assert torch.equal(output, expected_output), "Forward pass output does not match expected value"


# TODO: fix and re-enable
def skip_test_model_with_cache():
    # Predefined weights and biases for reproducibility
    emb__dim = 4
    seq_len = 3
    num_heads = 2
    batch_size = 1
    num_layers = 4

    query = generate_query_vector(batch_size, seq_len, CODEBOOKS)
    model_one_shot = get_model(d_model=emb__dim, num_heads=num_heads, num_layers=num_layers)
    model_one_shot.training = False
    model_incremental = get_model(d_model=emb__dim, num_heads=num_heads, num_layers=num_layers)
    model_incremental.training = False
    output_incremental_1 = model_incremental(query, use_cache=True, cache_index=None)
    next_token = get_token(output_incremental_1[:, :, -1, :])
    full_seq = torch.cat([query, next_token], dim=-1)

    output_incremental_2 = model_incremental(next_token, use_cache=True, cache_index=3)
    output_one_shot = model_one_shot(full_seq)

    output_incremental = torch.cat([output_incremental_1, output_incremental_2], dim=2)
    assert torch.allclose(output_incremental, output_one_shot, atol=1e-6), "Forward pass output does not match expected value"


# TODO: fix and re-enable
def skip_test_model_with_cache_last_element():
    emb_dim = 4
    seq_len = 3
    num_heads = 2
    batch_size = 1
    num_layers = 4

    query = generate_query_vector(batch_size, seq_len - 1, CODEBOOKS)
    model_one_shot = get_model(d_model=emb_dim, num_heads=num_heads, num_layers=num_layers)
    model_one_shot.training = False
    model_incremental = get_model(d_model=emb_dim, num_heads=num_heads, num_layers=num_layers)
    model_incremental.training = False
    
    output_incremental_1 = model_incremental(query, use_cache=True, cache_index=None)
    next_token = get_token(output_incremental_1[:, :, -1, :])
    full_seq = torch.cat([query, next_token], dim=-1)

    output_incremental_2 = model_incremental(next_token, use_cache=True, cache_index=2)
    output_one_shot = model_one_shot(full_seq)

    output_incremental = torch.cat([output_incremental_1, output_incremental_2], dim=2)
    assert torch.allclose(output_incremental, output_one_shot, atol=1e-6), "Forward pass output does not match expected value"


# TODO: fix and re-enable
def skip_test_model_with_sliding_window():
    """
    This tests that the output from the model when using a sliding window is different from the output of the model without it.
    It's difficult to predict how exactly the output would change, therefore we are basing our expectation of fixed values.
    """

    emb_dim = 2048
    seq_len = 512
    num_heads = 4
    batch_size = 1
    num_layers = 4
    sliding_window_size = 8

    query = generate_query_vector(batch_size, seq_len - 1, CODEBOOKS)
    model = get_model(d_model=emb_dim, num_heads=num_heads, num_layers=num_layers, sliding_window_size=sliding_window_size)
    model.training = False
    
    output = model(query)[:, :, -1]

    expected_output = torch.tensor(
        [[[-6.4006, 6.9212, -9.3461],
         [5.3698, -8.0366, -1.2782]]]
    )

    assert torch.allclose(expected_output, output, atol=1e-6), "Forward pass output does not match expected value"


def test_get_key_padding_mask():

    kpm = get_key_padding_mask(10, [2, 5, 10, 0])

    expected = torch.tensor([
        [False, False, True, True, True, True, True, True, True, True],
        [False, False, False, False, False, True, True, True, True, True],
        [False, False, False, False, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, True, True, True],
    ])

    assert torch.equal(kpm, expected)


def test_get_key_padding_mask_inverted():

    kpm = get_key_padding_mask(10, [2, 5, 10, 0], inverted=True)

    expected = torch.tensor([
        [True, True, True, True, True, True, True, True, False, False],
        [True, True, True, True, True, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, True, True, True],
    ])

    assert torch.equal(kpm, expected)


def test_generate_causal_sliding_mask():
    
    mask = generate_causal_sliding_mask(5, 2)

    expected = torch.tensor([
        [False, True, True, True, True],
        [False, False, True, True, True],
        [True, False, False, True, True],
        [True, True, False, False, True],
        [True, True, True, False, False]
    ])

    print(mask)

    assert torch.equal(mask, expected)
