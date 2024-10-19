import itertools
import time
from encodec.utils import save_audio
from gigmate.dataset.dataset import get_data_loader
from gigmate.model.codec import decode, get_codec

from gigmate.model.model import TransformerModel, get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.domain.prediction import complete_sequence
from gigmate.utils.constants import get_pad_token_id, get_params
from gigmate.utils.device import get_device, Device

NUMBER_OF_INPUT_TOKENS_FOR_PREDICTION = min(get_params()['max_seq_len'], 127)
NUM_OUTPUT_FILES = 5
EOS_TOKEN_ID = 2
SUBSET_OF_TEST_DATASET_NUMBER = 2
AUDIO_TO_GENERATE_LENGTH = 15


def test_model(model: TransformerModel, device: Device, data_loader, frame_rate: int) -> None:
    data_items = list(itertools.islice(iter(data_loader), SUBSET_OF_TEST_DATASET_NUMBER * NUM_OUTPUT_FILES, (SUBSET_OF_TEST_DATASET_NUMBER + 1) * NUM_OUTPUT_FILES))

    for i in list(range(0, NUM_OUTPUT_FILES)):
        output_file = f'/tmp/output_{i}.wav'
        print(f'Generating audio file output {i}:')
        # input_file = 'resources/test_generation.wav'
        input_file = data_items[i]['inputs'].full_track[0:1]
        print(f'Generating {AUDIO_TO_GENERATE_LENGTH} seconds of audio')
        start_time = time.perf_counter()
        output_sequence = complete_sequence(
            model,
            device,
            input_file,
            frame_rate=frame_rate,
            max_output_length_in_seconds=AUDIO_TO_GENERATE_LENGTH,
            padding_value=get_pad_token_id(),
            use_cache=True,
            show_progress=True,
        )
        end_time = time.perf_counter()
        print(f'Prediction took {end_time - start_time} seconds')
        output_tensor, sr = decode(output_sequence, device)
        end_time = time.perf_counter()
        print(f'Prediction + decoding took {end_time - start_time} seconds')
        save_audio(output_tensor.detach().cpu(), output_file, sample_rate=sr)
        print(f'Saved predicted file at {output_file}')


if __name__ == '__main__':
    device = get_device()
    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())
    data_loader = get_data_loader('validation')
    codec = get_codec(device)
    frame_rate = codec.config.frame_rate

    test_model(model, device, data_loader, frame_rate)
    
