import io
from miditok import TokSequence, MusicTokenizer
import torch
from tqdm import tqdm
from gigmate.audio_utils import generate_random_filename
from pretty_midi import PrettyMIDI
from gigmate.tokenizer import get_tokenizer, get_tokens_to_ids_dict

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

def predict_next_token(model: torch.nn.Module, input_sequence: torch.Tensor, temperature: float = DEFAULT_TEMPERATURE, next_token_idx: int = -1, forbidden_tokens: list[int] = []) -> int:
    with torch.inference_mode():
        outputs = model(input_sequence.unsqueeze(0))
        outputs = outputs.squeeze(0)[next_token_idx] # remove batch dimension and take only next token logits
        outputs = remove_forbidden_tokens(outputs, forbidden_tokens)    

    predicted_tokens = sample_from_logits(outputs, temperature)
    next_token = predicted_tokens # take last token
    return next_token

def get_length_in_seconds(tokenizer: MusicTokenizer, sequence: torch.Tensor) -> float:
    score = tokenizer.decode(sequence)
    bytes = score.dumps_midi()
    in_memory_file = io.BytesIO(bytes)
    midi = PrettyMIDI(in_memory_file)
    return midi.get_end_time()

def get_program_change_token(midi_program: int, tokenizer: MusicTokenizer) -> int:
    token = tokenizer.vocab[f'Program_{midi_program}']
    if token is None:
        raise Exception(f'Token for program {midi_program} not present in the tokenizer vocabulary')
    return token

def get_all_program_change_tokens(tokens_to_ids: dict[str, list[int]], except_for_program: int = None) -> list[int]:
    return [id for token, ids in tokens_to_ids.items() for id in ids if token.startswith('Program') and not token == f'Program_{except_for_program}']

def compute_output_sequence(
    model: torch.nn.Module,
    device: torch.device,
    tokenizer: MusicTokenizer,
    input_sequence: torch.Tensor,
    max_seq_len: int,
    midi_program: int = DEFAULT_MIDI_PROGRAM,
    verbose: bool = False,
    include_input: bool = True,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_output_length_in_seconds: float = DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
    show_progress: bool = False
) -> list[int]:

    def update_next_sequence(seq, next_token, max_len):
        if len(seq) == max_len:
            return torch.cat([seq[1:], next_token], 0).to(seq.device)
        else:
            return torch.cat([seq, next_token], 0).to(seq.device)

    def decode_and_print(tokenizer, note, tokens, generated_sequence_length: float):
        try:
            sequence = TokSequence(ids=[note.item()], are_ids_encoded=True)
            tokenizer.decode_token_ids(sequence)
            tokenizer.complete_sequence(sequence)
            tokens += sequence.tokens
            print(f'Length: {generated_sequence_length:.2f}s Tokens: {tokens}')

        except Exception as e:
            print('Error', e)

    def process_next_note_sequence(model, next_sequence, output_sequence, max_seq_len, temperature, forbidden_tokens: list[int] = []):
        next_token = predict_next_token(model, next_sequence, temperature, forbidden_tokens=forbidden_tokens)
        output_sequence = output_sequence + [next_token.item()]
        next_sequence = update_next_sequence(next_sequence, next_token, max_seq_len)
        return next_token, next_sequence, output_sequence
    
    def get_initial_next_sequence(initial_sequence, max_seq_len):
        length_to_keep = min(len(initial_sequence), max_seq_len)
        return initial_sequence[-length_to_keep:]

    model.eval()
    initial_sequence = input_sequence.copy()
    initial_sequence_length_in_seconds = get_length_in_seconds(tokenizer, initial_sequence) 
    
    if midi_program is not None:
        if verbose:
            print(f'Using midi program {midi_program}')
        current_program_change_token = get_program_change_token(midi_program, tokenizer)
        initial_sequence = initial_sequence + [current_program_change_token]

    output_sequence = initial_sequence.copy() if include_input else []
    next_sequence = torch.tensor(get_initial_next_sequence(initial_sequence, max_seq_len)).to(device)
    next_note = -1
    tokens = []

    loop = tqdm(range(max_output_tokens)) if show_progress else range(max_output_tokens)

    for _ in loop:
        if next_note == EOS_TOKEN_ID:
            break

        next_note, next_sequence, new_output_sequence = process_next_note_sequence(model, next_sequence, output_sequence, max_seq_len, temperature)
        
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
    device: torch.device,
    midi_file: str,
    tokenizer: MusicTokenizer,
    max_seq_len: int,
    verbose: bool = False,
    midi_program: int = DEFAULT_MIDI_PROGRAM,
    include_input: bool = True,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_output_length_in_seconds: float = DEFAULT_MAX_OUTPUT_LENGTH_IN_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:
    score = tokenizer.encode(midi_file)
    input_sequence = score.ids
    output_sequence = compute_output_sequence(model, device, tokenizer, input_sequence, max_seq_len=max_seq_len, midi_program=midi_program, verbose=verbose, include_input=include_input, max_output_tokens=max_output_tokens, max_output_length_in_seconds=max_output_length_in_seconds, temperature=temperature)
    midi_output = tokenizer.decode(output_sequence)
    temp_file = generate_random_filename(extension='.mid')
    midi_output.dump_midi(temp_file)
    return temp_file

if __name__ == '__main__':
    tokenizer = get_tokenizer()
    d = get_tokens_to_ids_dict(tokenizer)
    print('eos', d.get('BOS_None'))