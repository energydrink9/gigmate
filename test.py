import torch
from dataset import get_data_loaders
from model import get_model
from tokenizer import get_tokenizer
from miditok import TokSequence
from constants import get_params
from device import get_device

INPUT_TOKENS_COUNT = 250
OUTPUT_TOKENS_COUNT = 3
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

def predict_next_note(model, input_sequence, temperature=0, past_key_values=None):
    inp = input_sequence.unsqueeze(0)
    with torch.no_grad():
        outputs = model(inp)
        #logits, key_values = model(inp, past_key_values=past_key_values, use_cache=True)
    predicted_tokens = sample_from_logits(outputs[0], temperature)
    next_token = predicted_tokens[-1].squeeze()
    key_values = None
    return next_token, key_values

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
    key_values = None
    while i < OUTPUT_TOKENS_COUNT and next_note != 2:  # 2 is the end of sequence code
        next_note, key_values = predict_next_note(model, next_sequence, temperature=0.2, past_key_values=key_values)
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

    data_loader = get_data_loader()

    for i in range(NUM_OUTPUT_FILES):
        first_item = next(iter(data_loader))['input_ids']
        input_sequence = get_input_sequence(first_item).to(device)
        output_sequence = compute_output_sequence(model, tokenizer, input_sequence)

        create_midi_from_sequence(tokenizer, input_sequence, get_input_midi_file_name(i))
        create_midi_from_sequence(tokenizer, output_sequence, get_output_midi_file_name(i))

if __name__ == '__main__':
    device = get_device()
    model = get_model()
    model.to(device)
    test_model(model, device)