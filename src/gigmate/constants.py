RANDOM_SEED = 42
PAD_TOKEN_ID = 0

# Parameters
MAX_SEQ_LEN = 512
VOCAB_SIZE = 10000
D_MODEL = 512

NUM_LAYERS = 12
NUM_HEADS = 8
DROPOUT_RATE = 0.1

EPOCHS = 8
LEARNING_RATE = 0.00003
MAX_LEARNING_RATE = 0.001
STEP_SIZE_UP = 5000
GRADIENT_CLIP = 1.0

BATCH_SIZE = 16
ACCUMULATE_GRAD_BATCHES = 1

TRAINING_SET_SIZE = 1024 * 16
VALIDATION_SET_SIZE = 1024 * 2

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
    'max_learning_rate': MAX_LEARNING_RATE,
    'step_size_up': STEP_SIZE_UP,
    'accumulate_grad_batches': ACCUMULATE_GRAD_BATCHES,
    'gradient_clip': GRADIENT_CLIP,
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