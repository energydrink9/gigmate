import os
import numpy as np
import glob
from miditok import REMI, MusicTokenizer, TokenizerConfig
from gigmate.constants import get_params

SAMPLE_COUNT = 5000
TOKENIZER_MODEL_FILE = 'tokenizer_model'
TOKENIZER_CONFIGURATION_PATH = 'tokenizer.json'

def new_tokenizer():
    tokenizer = REMI(TokenizerConfig(use_programs=True, use_rests = True, use_tempos=True, use_time_signatures=True, use_chords=True))
    return tokenizer

def get_tokenizer():
    tokenizer = new_tokenizer()
    tokenizer = REMI(params = TOKENIZER_CONFIGURATION_PATH)
    if os.path.exists(TOKENIZER_MODEL_FILE):
        tokenizer._model.from_file(TOKENIZER_MODEL_FILE)
    print('Loaded tokenizer:')
    #print(f'Vocabulary: {tokenizer.vocab}')
    #print(f'Keys: {tokenizer._model.get_vocab().keys()}')
    print(f'Special tokens: {tokenizer.special_tokens}')
    print(f'Special tokens ids: {tokenizer.special_tokens_ids}')
    return tokenizer

def save_tokenizer(tokenizer, out_file: str):
    tokenizer._model.save(out_file)

def get_midi_files(directory: str):
    return glob.glob(os.path.join(directory, '**/*.mid'), recursive=True)

def train_tokenizer(mid_files_directory: str):
    paths_midis = list(get_midi_files(mid_files_directory))[:SAMPLE_COUNT]
    random_midis = np.random.choice(paths_midis, size=SAMPLE_COUNT, replace=False)
    # Learns the vocabulary with BPE
    # Ids are split per bars by default
    tokenizer = new_tokenizer()
    tokenizer.train(
        vocab_size=get_params()['vocab_size'],
        model="BPE",
        files_paths=random_midis,
    )
    return tokenizer

def get_tokens_to_ids_dict(tokenizer: MusicTokenizer):
    tokens_to_ids = dict()
    for byte, tokens in tokenizer._vocab_learned_bytes_to_tokens.items():
        id = tokenizer._model.token_to_id(byte)
        for token in tokens:
            if token in tokens_to_ids:
                tokens_to_ids[token] = tokens_to_ids[token] + [id]
            else:
                tokens_to_ids[token] = [id]

    return tokens_to_ids