from encodec.utils import save_audio
from gigmate.model.codec import decode, encode_file, get_codec

from gigmate.model.model import TransformerModel, get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.domain.prediction import complete_sequence
from gigmate.utils.constants import get_pad_token_id, get_params
from gigmate.utils.device import get_device

NUMBER_OF_INPUT_TOKENS_FOR_PREDICTION = min(get_params()['max_seq_len'], 127)
NUM_OUTPUT_FILES = 5
EOS_TOKEN_ID = 2
SUBSET_OF_TEST_DATASET_NUMBER = 2


def get_input_midi_file_name(i: int) -> str:
    return f'output/input_{i}.mid'


def get_output_midi_file_name(i: int) -> str:
    return f'output/output_{i}.mid'


def convert_to_midi(tokenizer, predicted_notes):
    return tokenizer.decode(predicted_notes)


def test_model(model: TransformerModel, device: str, data_loader):
    # data_items = list(itertools.islice(iter(data_loader), SUBSET_OF_TEST_DATASET_NUMBER * NUM_OUTPUT_FILES, (SUBSET_OF_TEST_DATASET_NUMBER + 1) * NUM_OUTPUT_FILES))

    files = []
    for i in list(range(0, NUM_OUTPUT_FILES)):
        output_file = f'output/output_{i}.wav'
        print(f'Generating audio file output {i}:')
        input_file = 'resources/test_generation.wav'
        codec = get_codec().to(device)
        input_sequence, frame_rate = encode_file(input_file, device)

        files.append({'name': f'input_{i}', 'file': input_file})
        
        output_sequence = complete_sequence(
            model,
            device,
            input_sequence[0],
            frame_rate=frame_rate,
            max_output_length_in_seconds=1,
            padding_value=get_pad_token_id(),
            use_cache=True,
            show_progress=True
        )
        output_tensor, sr = decode(output_sequence, device)
        save_audio(output_tensor, output_file, sample_rate=sr)

        files.append({'name': f'output_{i}', 'file': output_file})

    return files


if __name__ == '__main__':
    device = get_device()
    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())
    # data_loader = get_data_loader('validation')

    test_model(model, device, None)
    
