from typing import List

# Hyperparameters

MAX_SEQ_LEN = 1024
MAX_DECODER_SEQ_LEN = 512
SLIDING_WINDOW_SIZE = 512
VOCAB_SIZE = 2048
D_MODEL = 1024
CODEBOOKS = 4

ENCODER_LAYERS = 20
DECODER_LAYERS = 28
NUM_HEADS = 16
DROPOUT_RATE = 0.1

EPOCHS = 40
LEARNING_RATE = 1e-4
MAX_LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-7
GRADIENT_CLIP = 1.0

BATCH_SIZE = 12
ACCUMULATE_GRAD_BATCHES = 1

TRAINING_SET_SIZE = 1.0
VALIDATION_SET_SIZE = 1.0

RANDOM_SEED = 42
USE_ALIBI = False
USE_CUSTOM_MODEL = False
USE_SLIDING_WINDOW = True

# Constants
PAD_TOKEN_ID = VOCAB_SIZE - 3  # 2045
SOS_TOKEN_ID = VOCAB_SIZE - 2  # 2046
EOS_TOKEN_ID = VOCAB_SIZE - 1  # 2047

CLEARML_PROJECT_NAME = 'GigMate'
CLEARML_DATASET_PROJECT_NAME = 'stem_continuation_dataset_generator'
CLEARML_DATASET_NAME = 'stem_continuation_dataset'
CLEARML_DATASET_TAGS = ['stem-drum']
CLEARML_DATASET_VERSION = '1.0.0'

PARAMS = {
    'vocab_size': VOCAB_SIZE,
    'encoder_layers': ENCODER_LAYERS,
    'decoder_layers': DECODER_LAYERS,
    'd_model': D_MODEL,
    'codebooks': CODEBOOKS,
    'num_heads': NUM_HEADS,
    'dff': D_MODEL * 6,
    'dropout_rate': DROPOUT_RATE,
    'epochs': EPOCHS,
    'max_seq_len': MAX_SEQ_LEN,
    'max_decoder_seq_len': MAX_DECODER_SEQ_LEN,
    'sliding_window_size': SLIDING_WINDOW_SIZE,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'max_learning_rate': MAX_LEARNING_RATE,
    'min_learning_rate': MIN_LEARNING_RATE,
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


def get_special_tokens() -> List[int]:
    return [get_start_of_sequence_token_id(), get_end_of_sequence_token_id(), get_pad_token_id()]


def get_params():
    return PARAMS


def get_clearml_project_name() -> str:
    return CLEARML_PROJECT_NAME


def get_clearml_dataset_project_name() -> str:
    return CLEARML_DATASET_PROJECT_NAME


def get_clearml_dataset_name() -> str:
    return CLEARML_DATASET_NAME


def get_clearml_dataset_tags() -> List[str]:
    return CLEARML_DATASET_TAGS


def get_clearml_dataset_version() -> str:
    return CLEARML_DATASET_VERSION


def get_use_alibi() -> bool:
    return USE_ALIBI


def get_use_custom_model() -> bool:
    return USE_CUSTOM_MODEL