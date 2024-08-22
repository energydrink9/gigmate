RANDOM_SEED = 42
MAX_SEQ_LEN = 512
VOCAB_SIZE = 10000
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT_RATE = 0.1
EPOCHS = 10
BATCH_SIZE = 16

PARAMS = {
    'vocab_size': VOCAB_SIZE,
    'num_layers': NUM_LAYERS,
    'd_model': D_MODEL,
    'num_heads': NUM_HEADS,
    'dff': D_MODEL * 4,
    'dropout_rate': DROPOUT_RATE,
    'epochs': EPOCHS,
    'max_seq_len': MAX_SEQ_LEN,
    'batch_size': BATCH_SIZE
}

def get_random_seed() -> int:
    return RANDOM_SEED

def get_params():
    return PARAMS
