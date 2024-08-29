RANDOM_SEED = 42
PAD_TOKEN_ID = 0

# Parameters
MAX_SEQ_LEN = 64
VOCAB_SIZE = 10000
D_MODEL = 512

NUM_LAYERS = 32
NUM_HEADS = 32
DROPOUT_RATE = 0.1

EPOCHS = 8
LEARNING_RATE = 0.000003

BATCH_SIZE = 32
ACCUMULATE_GRAD_BATCHES = 1

TRAINING_SET_SIZE = 1024 * 8#1.0
VALIDATION_SET_SIZE = 1024#1.0

CLEARML_PROJECT_NAME = 'GigMate'
CLEARML_DATASET_NAME = 'LakhMidiClean'
CLEARML_DATASET_VERSION = '1.0.17'

PARAMS = {
    'vocab_size': VOCAB_SIZE,
    'num_layers': NUM_LAYERS,
    'd_model': D_MODEL,
    'num_heads': NUM_HEADS,
    'dff': D_MODEL * 4,
    'dropout_rate': DROPOUT_RATE,
    'epochs': EPOCHS,
    'max_seq_len': MAX_SEQ_LEN,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'accumulate_grad_batches': ACCUMULATE_GRAD_BATCHES,
    'training_set_size': TRAINING_SET_SIZE,
    'validation_set_size': VALIDATION_SET_SIZE,
}

def get_random_seed() -> int:
    return RANDOM_SEED

def get_pad_token_id() -> int:
    return PAD_TOKEN_ID

def get_params():
    return PARAMS

def get_clearml_project_name() -> str:
    return CLEARML_PROJECT_NAME

def get_clearml_dataset_name() -> str:
    return CLEARML_DATASET_NAME

def get_clearml_dataset_version() -> str:
    return CLEARML_DATASET_VERSION