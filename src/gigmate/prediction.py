from miditok import TokSequence
import torch
from tqdm import tqdm
from gigmate.audio_utils import generate_random_filename
from miditok.utils import num_bar_pos

DEFAULT_OUTPUT_TOKENS = 100
EOS_TOKEN_ID = 0
DEFAULT_TEMPERATURE = 0.3

def sample_from_logits(logits, temperature):
    # If temp is 0 then next_token is the argmax of logits
    if temperature == 0.0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    # If temp is not 0 then next_token is sampled out of logits
    else:
        logits = logits / temperature
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
    return next_token

def predict_next_token(model, input_sequence, temperature=DEFAULT_TEMPERATURE, next_note_idx = -1):
    with torch.inference_mode():
        outputs = model(input_sequence.unsqueeze(0))
        outputs = outputs.squeeze(0) # remove batch dimension
    predicted_tokens = sample_from_logits(outputs, temperature)
    next_token = predicted_tokens[next_note_idx] # take last token
    return next_token

def compute_output_sequence(model, tokenizer, input_sequence, max_seq_len, verbose=False, include_input=True, output_tokens=DEFAULT_OUTPUT_TOKENS, temperature=DEFAULT_TEMPERATURE):
    def update_next_sequence(seq, note, max_len):
        if len(seq) == max_len:
            return torch.cat([seq[1:], note], 0).to(seq.device)
        else:
            return torch.cat([seq, note], 0).to(seq.device)

    def decode_and_print(tokenizer, note, tokens):
        try:
            sequence = TokSequence(ids=[note.item()], are_ids_encoded=True)
            tokenizer.decode_token_ids(sequence)
            tokenizer.complete_sequence(sequence)
            tokens += sequence.tokens
            print(f'{tokens}', end='\r')
        except Exception as e:
            print('Error', e)

    def process_next_note_sequence(model, next_sequence, output_sequence, max_seq_len, temperature):
        next_token = predict_next_token(model, next_sequence, temperature)
        output_sequence = torch.cat([output_sequence, next_token], 0).to(next_sequence.device)
        next_sequence = update_next_sequence(next_sequence, next_token, max_seq_len)
        return next_token, next_sequence, output_sequence
    
    def get_initial_next_sequence(initial_sequence, max_seq_len, device):
        length_to_keep = min(len(initial_sequence), max_seq_len)
        return initial_sequence[-length_to_keep:].to(device)

    model.eval()
    initial_sequence = input_sequence.clone().detach().to(input_sequence.device)

    if len(initial_sequence) == 0:
        if verbose:
            print('No initial sequence')
        return torch.tensor([], dtype=input_sequence.dtype).to(input_sequence.device)

    output_sequence = initial_sequence if include_input else torch.tensor([], dtype=input_sequence.dtype).to(input_sequence.device)
    next_sequence = get_initial_next_sequence(initial_sequence, max_seq_len, input_sequence.device)
    next_note = -1
    tokens = []

    loop = tqdm(range(output_tokens)) if verbose else range(output_tokens)

    for _ in loop:
        if next_note == EOS_TOKEN_ID:
            break

        next_note, next_sequence, output_sequence = process_next_note_sequence(model, next_sequence, output_sequence, max_seq_len, temperature)
        if verbose:
            print(f'Next sequence: {next_sequence}')
            decode_and_print(tokenizer, next_note, tokens)

    if verbose:
        print(f'output: {output_sequence}')

    return output_sequence.detach().cpu().numpy()

def complete_midi(model, device, midi_file, tokenizer, max_seq_len, verbose=False, include_input=True, output_tokens=DEFAULT_OUTPUT_TOKENS, temperature=DEFAULT_TEMPERATURE):
    score = tokenizer.encode(midi_file)
    input_sequence = torch.tensor(score.ids).to(device)
    output_sequence = compute_output_sequence(model, tokenizer, input_sequence, max_seq_len=max_seq_len, verbose=verbose, include_input=include_input, output_tokens=output_tokens, temperature=temperature)
    midi_output = tokenizer.decode(output_sequence)
    temp_file = generate_random_filename(extension='.mid')
    midi_output.dump_midi(temp_file)
    return temp_file