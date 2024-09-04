import itertools
import torch
from tqdm import tqdm
from gigmate.dataset import get_data_loader
from gigmate.model import get_latest_model_checkpoint
from gigmate.tokenizer import get_tokenizer
from miditok import TokSequence
from gigmate.constants import get_params
from gigmate.device import get_device

INPUT_TOKENS_COUNT = min(get_params()['max_seq_len'], 127)
OUTPUT_TOKENS_COUNT = 1000
NUM_OUTPUT_FILES = 6
EOS_TOKEN_ID = 2
SET = 2

def get_input_midi_file_name(i: int) -> str:
    return f'output/input_{i}.mid'

def get_output_midi_file_name(i: int) -> str:
    return f'output/output_{i}.mid'

def sample_from_logits(logits, temperature):
    # If temp is 0 then next_token is the argmax of logits
    if temperature == 0.0:
        next_token = torch.argmax(logits, dim=-1)
    # If temp is not 0 then next_token is sampled out of logits
    else:
        logits = logits / temperature
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
    return next_token

def predict_next_note(model, input_sequence, next_note_idx = -1, temperature=0):
    with torch.inference_mode():
        outputs = model(input_sequence.unsqueeze(0))
        outputs = outputs.squeeze(0) # remove batch dimension
    predicted_tokens = sample_from_logits(outputs, temperature)
    next_token = predicted_tokens[next_note_idx] # take last token
    return next_token

def convert_to_midi(tokenizer, predicted_notes):
    return tokenizer.decode(predicted_notes)

def get_sequence(prediction):
    token_ids = prediction.tolist()
    sequence = TokSequence(ids=token_ids, are_ids_encoded=True)
    return sequence

def create_midi_from_sequence(tokenizer, sequence, out_file):
    sequence = get_sequence(sequence)
    midi_path = convert_to_midi(tokenizer, sequence)
    midi_path.dump_midi(out_file)
    print(f'created midi file: {out_file}')
    return out_file

def get_input_sequence(batch):
    return batch[0][:INPUT_TOKENS_COUNT]

def compute_output_sequence(model, tokenizer, input_sequence, max_seq_len, verbose=False, include_input=True, output_tokens=OUTPUT_TOKENS_COUNT):
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

    def process_next_note_sequence(model, next_sequence, output_sequence, max_seq_len):
        next_note = predict_next_note(model, next_sequence, temperature=0.3)
        output_sequence = torch.cat([output_sequence, next_note], 0).to(next_sequence.device)
        next_sequence = update_next_sequence(next_sequence, next_note, max_seq_len)
        return next_note, next_sequence, output_sequence
    
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

    for _ in tqdm(range(output_tokens)):
        if next_note == EOS_TOKEN_ID:
            break

        next_note, next_sequence, output_sequence = process_next_note_sequence(model, next_sequence, output_sequence, max_seq_len)
        if verbose:
            print(f'Next sequence: {next_sequence}')
            decode_and_print(tokenizer, next_note, tokens)

    if verbose:
        print(f'output: {output_sequence}')

    return output_sequence.detach().cpu().numpy()

def test_model(model, device, data_loader):
    max_seq_len = get_params()['max_seq_len']
    tokenizer = get_tokenizer()
    data_items = list(itertools.islice(iter(data_loader), SET * NUM_OUTPUT_FILES, (SET + 1) * NUM_OUTPUT_FILES))

    files = []
    for i in list(range(0, NUM_OUTPUT_FILES)):
        print(f'Generating MIDI output {i}:')
        next_item = data_items[i]['input_ids'].to(device)
        input_sequence = get_input_sequence(next_item).to(device)
        input_file = create_midi_from_sequence(tokenizer, input_sequence, get_input_midi_file_name(i))
        files.append({ 'name': f'input_{i}', 'file': input_file })

        output_sequence = compute_output_sequence(model, tokenizer, input_sequence, max_seq_len)
        output_file = create_midi_from_sequence(tokenizer, output_sequence, get_output_midi_file_name(i))
        files.append({ 'name': f'output_{i}', 'file': output_file })

    return files

def complete_midi(model, midi_file, tokenizer, max_seq_len, start_after_idx = -1, output_file_name = 'output'):
    score = tokenizer.encode(midi_file)
    input_sequence = torch.tensor(score.ids[0: start_after_idx]).to(device)
    create_midi_from_sequence(tokenizer, input_sequence, get_input_midi_file_name(output_file_name))
    return compute_output_sequence(model, tokenizer, input_sequence, max_seq_len)

if __name__ == '__main__':
    device = get_device()
    model = get_latest_model_checkpoint(device)
    
    data_loader = get_data_loader('validation')

    test_model(model, device, data_loader)
    
