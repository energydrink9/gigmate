import itertools
import torch
from gigmate.dataset import get_data_loaders
from gigmate.model import get_model
from gigmate.tokenizer import get_tokenizer
from miditok import TokSequence
from gigmate.constants import get_params
from gigmate.device import get_device

INPUT_TOKENS_COUNT = min(get_params()['max_seq_len'], 256)
OUTPUT_TOKENS_COUNT = 5000
NUM_OUTPUT_FILES = 5
EOS_TOKEN_ID = 2

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
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze()
    return next_token

def predict_next_note(model, input_sequence, temperature=0):
    with torch.no_grad():
        outputs = model(input_sequence)
        outputs = outputs.squeeze() # remove batch dimension
    predicted_tokens = sample_from_logits(outputs, temperature)
    next_token = predicted_tokens[-1] # take last token
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
    print(f'created midi file: {out_file}')
    return out_file

def get_input_sequence(batch):
    return torch.cat([torch.tensor([0]).to(batch.device), batch[0][:INPUT_TOKENS_COUNT]]) # TODO: remove 0 that is added for the start token

def compute_output_sequence(model, tokenizer, input_sequence, verbose=False):
    output_sequence = input_sequence.clone().detach().to(input_sequence.device)
    length_to_keep = min(len(output_sequence), get_params()['max_seq_len'])
    next_sequence = output_sequence[-length_to_keep:].to(device)

    next_note = -1
    i = 0
    while i < OUTPUT_TOKENS_COUNT and next_note != EOS_TOKEN_ID:
        next_note = predict_next_note(model, next_sequence, temperature=0.3)
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
        output_sequence = torch.cat([output_sequence, next_note.unsqueeze(0)], 0).to(device)
        next_sequence = torch.cat([next_sequence[1:], next_note.unsqueeze(0)], 0).to(device)
        i += 1

    if verbose:
        print(f'output: {output_sequence}')

    return output_sequence

def test_model(model, device, data_loader):
    tokenizer = get_tokenizer()
    data_items = list(itertools.islice(iter(data_loader), NUM_OUTPUT_FILES))

    files = []
    for i in list(range(NUM_OUTPUT_FILES)):
        next_item = data_items[i]['input_ids'].to(device)
        input_sequence = get_input_sequence(next_item).to(device)
        output_sequence = compute_output_sequence(model, tokenizer, input_sequence)

        input_file = create_midi_from_sequence(tokenizer, input_sequence, get_input_midi_file_name(i))
        output_file = create_midi_from_sequence(tokenizer, output_sequence, get_output_midi_file_name(i))
        files.append({ 'name': f'input_{i}', 'file': input_file })
        files.append({ 'name': f'output_{i}', 'file': output_file })

    return files

if __name__ == '__main__':
    device = get_device()
    model = get_model()
    model.load_state_dict(torch.load('output/gigmate.weights', map_location=torch.device(device), weights_only=True), strict=False)
    model.to(device)

    data_loader = get_data_loader()
    test_model(model, device, data_loader)

