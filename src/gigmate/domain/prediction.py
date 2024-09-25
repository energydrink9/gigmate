from typing import Optional, cast
from miditok import TokSequence, MusicTokenizer
import torch
from tqdm import tqdm
from gigmate.utils.audio_utils import calculate_score_length_in_seconds, generate_random_filename
from gigmate.model.tokenizer import get_tokenizer, get_tokens_to_ids_dict
from gigmate.utils.constants import get_pad_token_id

DEFAULT_MAX_OUTPUT_TOKENS = 1000
EOS_TOKEN_ID = 0
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS = 20
DEFAULT_MIDI_PROGRAM = None

def sample_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # If temp is 0 then next_token is the argmax of logits
    if temperature == 0.0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    # If temp is not 0 then next_token is sampled out of logits
    else:
        logits = logits / temperature
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

    return next_token

def remove_forbidden_tokens(outputs: torch.Tensor, forbidden_tokens: list[int]):
    outputs[forbidden_tokens] = float('-inf')
    return outputs

def predict_next_token(model: torch.nn.Module, input_sequence: torch.Tensor, current_token_index: int, incremental: bool, temperature: float = DEFAULT_TEMPERATURE, forbidden_tokens: list[int] = []) -> int:
    
    with torch.inference_mode():
        input = input_sequence.unsqueeze(0)
        outputs = model(input, use_cache=True, current_token_index=current_token_index if incremental else None)
        outputs = outputs.squeeze(0)[-1 if incremental else current_token_index] # remove batch dimension and take only next token logits
        outputs = remove_forbidden_tokens(outputs, forbidden_tokens)

    predicted_tokens = sample_from_logits(outputs, temperature)
    next_token = predicted_tokens # take last token
    return cast(int, next_token.item())

def get_length_in_seconds(tokenizer: MusicTokenizer, sequence: list[int]) -> float:
    score = tokenizer.decode(sequence)
    return calculate_score_length_in_seconds(score)

def get_program_change_token(midi_program: int, tokenizer: MusicTokenizer) -> int:
    token = tokenizer.vocab[f'Program_{midi_program}']
    if token is None:
        raise Exception(f'Token for program {midi_program} not present in the tokenizer vocabulary')
    return token

def get_all_program_change_tokens(tokens_to_ids: dict[str, list[int]], except_for_program: Optional[int] = None) -> list[int]:
    return [id for token, ids in tokens_to_ids.items() for id in ids if token.startswith('Program') and not token == f'Program_{except_for_program}']

def complete_sequence(
    model: torch.nn.Module,
    device: str,
    tokenizer: MusicTokenizer,
    input_sequence: list[int],
    max_seq_len: int,
    midi_program: Optional[int] = DEFAULT_MIDI_PROGRAM,
    verbose: bool = False,
    include_input: bool = True,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_output_length_in_seconds: float = DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
    show_progress: bool = False
) -> list[int]:
    
    def update_next_sequence(previous_next_sequence, current_token, max_seq_len, current_token_index: int):
        current_token_tensor = torch.tensor([current_token]).to(previous_next_sequence.device)

        if current_token_index == max_seq_len - 1:
            return torch.cat([previous_next_sequence[1:], current_token_tensor], 0).to(previous_next_sequence.device)
        else:
            previous_next_sequence[current_token_index + 1] = current_token
            return previous_next_sequence

    def decode_and_print(tokenizer: MusicTokenizer, note: int, tokens, generated_sequence_length: float):
        try:
            sequence = TokSequence(ids=[note], are_ids_encoded=True)
            tokenizer.decode_token_ids(sequence)
            tokenizer.complete_sequence(sequence)
            tokens += sequence.tokens
            print(f'Length: {generated_sequence_length:.2f}s Tokens: {tokens}')

        except Exception as e:
            print('Error', e)

    def process_next_note_sequence(model, next_sequence, current_token_index: int, output_sequence, max_seq_len, temperature, incremental: bool, forbidden_tokens: list[int] = []):
        next_token = predict_next_token(model, next_sequence, current_token_index, incremental, temperature, forbidden_tokens=forbidden_tokens)
        output_sequence = output_sequence + [next_token]
        next_sequence = update_next_sequence(next_sequence, next_token, max_seq_len, current_token_index)
        return next_token, next_sequence, output_sequence
    
    def pad(sequence, max_len):
        return sequence + [get_pad_token_id()] * (max_len - len(sequence))

    def get_initial_next_sequence(initial_sequence, max_seq_len):
        length_to_keep = min(len(initial_sequence), max_seq_len)
        return pad(initial_sequence[-length_to_keep:], max_seq_len)

    model.reset_kv_cache()
    model.eval()
    initial_sequence = input_sequence.copy()
    initial_sequence_length_in_seconds = get_length_in_seconds(tokenizer, initial_sequence)
    
    if midi_program is not None:
        if verbose:
            print(f'Using midi program {midi_program}')
        current_program_change_token = get_program_change_token(midi_program, tokenizer)
        initial_sequence = initial_sequence + [current_program_change_token]

    output_sequence = initial_sequence.copy() if include_input else []
    initial_token_index = len(output_sequence) - 1
    next_sequence = torch.tensor(get_initial_next_sequence(initial_sequence, max_seq_len)).to(device)

    next_note = -1
    tokens: list[int] = []

    loop = tqdm(range(max_output_tokens)) if show_progress else range(max_output_tokens)

    for iteration in loop:
        if next_note == EOS_TOKEN_ID:
            break
        
        current_token_index = min(initial_token_index + iteration, max_seq_len - 1)
        next_note, next_sequence, new_output_sequence = process_next_note_sequence(model, next_sequence, current_token_index, output_sequence, max_seq_len, temperature, incremental=iteration != 0)
        
        new_output_sequence_length_in_seconds = get_length_in_seconds(tokenizer, new_output_sequence) if include_input else get_length_in_seconds(tokenizer, initial_sequence + output_sequence)
        generated_sequence_length = new_output_sequence_length_in_seconds - initial_sequence_length_in_seconds

        if generated_sequence_length > max_output_length_in_seconds:
            break

        output_sequence = new_output_sequence

        if verbose:
            decode_and_print(tokenizer, next_note, tokens, generated_sequence_length)

    return output_sequence

def complete_midi(
    model: torch.nn.Module,
    device: str,
    midi_file: str,
    tokenizer: MusicTokenizer,
    max_seq_len: int,
    verbose: bool = False,
    midi_program: Optional[int] = DEFAULT_MIDI_PROGRAM,
    include_input: bool = True,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_output_length_in_seconds: float = DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:
    score = tokenizer.encode(midi_file)
    input_sequence = score.ids
    output_sequence = complete_sequence(model, device, tokenizer, input_sequence, max_seq_len=max_seq_len, midi_program=midi_program, verbose=verbose, include_input=include_input, max_output_tokens=max_output_tokens, max_output_length_in_seconds=max_output_length_in_seconds, temperature=temperature)
    midi_output = tokenizer.decode(output_sequence)
    temp_file = generate_random_filename(extension='.mid')
    midi_output.dump_midi(temp_file)
    return temp_file
