import itertools

from miditok import TokSequence
from gigmate.dataset import get_data_loader
from gigmate.model import get_model
from gigmate.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.prediction import compute_output_sequence
from gigmate.tokenizer import get_tokenizer
from gigmate.constants import get_params
from gigmate.device import get_device

NUMBER_OF_INPUT_TOKENS_FOR_PREDICTION = min(get_params()['max_seq_len'], 127)
OUTPUT_TOKENS_COUNT = 1000
NUM_OUTPUT_FILES = 6
EOS_TOKEN_ID = 2
SUBSET_OF_TEST_DATASET_NUMBER = 2

def get_input_midi_file_name(i: int) -> str:
    return f'output/input_{i}.mid'

def get_output_midi_file_name(i: int) -> str:
    return f'output/output_{i}.mid'

def get_input_sequence(batch):
    return batch[0][:NUMBER_OF_INPUT_TOKENS_FOR_PREDICTION]

def get_sequence(prediction):
    token_ids = prediction.tolist()
    sequence = TokSequence(ids=token_ids, are_ids_encoded=True)
    return sequence

def convert_to_midi(tokenizer, predicted_notes):
    return tokenizer.decode(predicted_notes)

def create_midi_from_sequence(tokenizer, sequence, out_file):
    sequence = get_sequence(sequence)
    midi_path = convert_to_midi(tokenizer, sequence)
    midi_path.dump_midi(out_file)
    print(f'created midi file: {out_file}')
    return out_file

def test_model(model, device, data_loader):
    max_seq_len = get_params()['max_seq_len']
    tokenizer = get_tokenizer()
    data_items = list(itertools.islice(iter(data_loader), SUBSET_OF_TEST_DATASET_NUMBER * NUM_OUTPUT_FILES, (SUBSET_OF_TEST_DATASET_NUMBER + 1) * NUM_OUTPUT_FILES))

    files = []
    for i in list(range(0, NUM_OUTPUT_FILES)):
        print(f'Generating MIDI output {i}:')
        next_item = data_items[i]['input_ids'].to(device)
        input_sequence = get_input_sequence(next_item).to(device)
        input_file = create_midi_from_sequence(tokenizer, input_sequence, get_input_midi_file_name(i))
        files.append({ 'name': f'input_{i}', 'file': input_file })

        output_sequence = compute_output_sequence(model, device, tokenizer, input_sequence, max_seq_len=max_seq_len)
        output_file = create_midi_from_sequence(tokenizer, output_sequence, get_output_midi_file_name(i))
        files.append({ 'name': f'output_{i}', 'file': output_file })

    return files

if __name__ == '__main__':
    device = get_device()
    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())    
    data_loader = get_data_loader('validation')

    test_model(model, device, data_loader)
    
