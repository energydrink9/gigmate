# Hyperparameters
MAX_SEQ_LEN = 1024 * 4
MAX_DECODER_SEQ_LEN = 512
SLIDING_WINDOW_SIZE = 256
VOCAB_SIZE = 2048 + 3
D_MODEL = 512
CODEBOOKS = 4

ENCODER_LAYERS = 12
DECODER_LAYERS = 8
NUM_HEADS = 16
DROPOUT_RATE = 0.1

EPOCHS = 20
LEARNING_RATE = 0.0001
MAX_LEARNING_RATE = 0.001
STEP_SIZE_UP = 300
GRADIENT_CLIP = 1.0

BATCH_SIZE = 6
ACCUMULATE_GRAD_BATCHES = 1

TRAINING_SET_SIZE = 1.0
VALIDATION_SET_SIZE = 1.0

RANDOM_SEED = 42

# Constants
PAD_TOKEN_ID = VOCAB_SIZE - 3  # 2048
SOS_TOKEN_ID = VOCAB_SIZE - 2  # 2049
EOS_TOKEN_ID = VOCAB_SIZE - 1  # 2050

CLEARML_PROJECT_NAME = 'GigMate'
CLEARML_DATASET_NAME = 'SoundStripe-4k'
CLEARML_DATASET_VERSION = '1.0.0'
CLEARML_DATASET_TRAINING_NAME = 'SoundStripe'
CLEARML_DATASET_TRAINING_VERSION = '1.0.0'

PARAMS = {
    'vocab_size': VOCAB_SIZE,
    'encoder_layers': ENCODER_LAYERS,
    'decoder_layers': DECODER_LAYERS,
    'd_model': D_MODEL,
    'codebooks': CODEBOOKS,
    'num_heads': NUM_HEADS,
    'dff': D_MODEL * 4,
    'dropout_rate': DROPOUT_RATE,
    'epochs': EPOCHS,
    'max_seq_len': MAX_SEQ_LEN,
    'max_decoder_seq_len': MAX_DECODER_SEQ_LEN,
    'sliding_window_size': SLIDING_WINDOW_SIZE,
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


def get_start_of_sequence_token_id() -> int:
    return SOS_TOKEN_ID


def get_end_of_sequence_token_id() -> int:
    return EOS_TOKEN_ID


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


def get_clearml_dataset_training_name() -> str:
    return CLEARML_DATASET_TRAINING_NAME


def get_clearml_dataset_training_version() -> str:
    return CLEARML_DATASET_TRAINING_VERSION
