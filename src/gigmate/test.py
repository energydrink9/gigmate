import itertools
import torch
from gigmate.dataset import get_data_loaders
from gigmate.model import get_model
from gigmate.tokenizer import get_tokenizer
from miditok import TokSequence
from gigmate.constants import get_params
from gigmate.device import get_device

INPUT_TOKENS_COUNT = 250
OUTPUT_TOKENS_COUNT = 5000
NUM_OUTPUT_FILES = 5

def get_input_midi_file_name(i: int) -> str:
    return f'output/input_{i}.mid'

def get_output_midi_file_name(i: int) -> str:
    return f'output/output_{i}.mid'

def sample_from_logits(logits, temperature):
    # If temp is 0 then next_token is the argmax of logits
    if temperature == 0.0:
        next_token = torch.argmax(logits, dim=1)
    
    # If temp is not 0 then next_token is sampled out of logits
    else:
        logits = logits / temperature
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
    return next_token

def predict_next_note(model, input_sequence, temperature=0):
    inp = input_sequence.unsqueeze(0)
    with torch.no_grad():
        outputs = model(inp)
    predicted_tokens = sample_from_logits(outputs[0], temperature)
    next_token = predicted_tokens.squeeze()
    return next_token

def get_data_loader():
    _, validation_loader, _ = get_data_loaders()
    return validation_loader

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

def get_input_sequence(batch):
    return batch[0][INPUT_TOKENS_COUNT:]

def compute_output_sequence(model, tokenizer, input_sequence, verbose=False):
    output_sequence = input_sequence.clone().detach().to(input_sequence.device)
    next_sequence = output_sequence[-min(len(output_sequence), get_params()['max_seq_len']) - 1:]

    next_note = -1
    i = 0
    while i < OUTPUT_TOKENS_COUNT and next_note != 2:  # 2 is the end of sequence code
        next_note = predict_next_note(model, next_sequence, temperature=0.2)
        meaning = ''
        try:
            sequence = TokSequence(ids=[next_note.item()], are_ids_encoded=True)
            tokenizer.decode_token_ids(sequence)
            tokenizer.complete_sequence(sequence)
            meaning = sequence.tokens
        except Exception as e:
            print('Error', e)
        if verbose:
            print(f'token {i}: {next_note}, meaning: {meaning}')
        output_sequence = torch.cat([output_sequence, next_note.unsqueeze(0)], 0)
        next_sequence = torch.cat([next_sequence[1:], next_note.unsqueeze(0)], 0)
        i += 1

    if verbose:
        print(f'output: {output_sequence}')

    return output_sequence

def test_model(model, device):
    tokenizer = get_tokenizer()

    buffer_size = 1000
    data_loader = get_data_loader()
    data_items = list(itertools.islice(iter(data_loader), NUM_OUTPUT_FILES * buffer_size))

    for i in range(NUM_OUTPUT_FILES):
        next_item = data_items[i * buffer_size]['input_ids']
        input_sequence = get_input_sequence(next_item).to(device)
        output_sequence = compute_output_sequence(model, tokenizer, input_sequence)

        create_midi_from_sequence(tokenizer, input_sequence, get_input_midi_file_name(i))
        create_midi_from_sequence(tokenizer, output_sequence, get_output_midi_file_name(i))

if __name__ == '__main__':
    device = get_device()
    model = get_model()
    model.to(device)
    test_model(model, device)