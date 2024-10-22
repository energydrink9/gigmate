import itertools
import time
from encodec.utils import save_audio
from gigmate.dataset.dataset import get_data_loader
from gigmate.model.codec import decode, get_codec

from gigmate.model.model import TransformerModel, get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.domain.prediction import complete_sequence
from gigmate.utils.constants import get_pad_token_id, get_special_tokens
from gigmate.utils.device import get_device, Device
from gigmate.utils.sequence_utils import cut_sequence, remove_special_tokens, revert_interleaving

NUM_OUTPUT_FILES = 5
SUBSET_OF_TEST_DATASET_NUMBER = 0
AUDIO_TO_GENERATE_LENGTH = 4
# NUMBER_OF_SECONDS_FOR_PREDICTION = 10


def test_model(model: TransformerModel, device: Device, data_loader, frame_rate: int) -> None:
    data_items = list(itertools.islice(iter(data_loader), SUBSET_OF_TEST_DATASET_NUMBER * NUM_OUTPUT_FILES, (SUBSET_OF_TEST_DATASET_NUMBER + 1) * NUM_OUTPUT_FILES))

    for i in list(range(0, NUM_OUTPUT_FILES)):
        input_file = f'/tmp/input_{i}.wav'
        stem_file = f'/tmp/stem_{i}.wav'
        output_file = f'/tmp/output_{i}.wav'
        
        print(f'Generating audio file output {i}:')
        # input_file = 'resources/test_generation.wav'

        input_sequence = data_items[i].inputs.full_track[:1, :, :]
        input_sequence = revert_interleaving(input_sequence)
        input_sequence = cut_sequence(input_sequence, data_items[i].sequence_lengths.full_track[0])
        input_sequence = remove_special_tokens(input_sequence, get_special_tokens())
        input_tensor, sr = decode(input_sequence, device)
        save_audio(input_tensor.detach().cpu(), input_file, sample_rate=sr)

        stem_sequence = data_items[i].inputs.stem[:1, :, :]
        stem_sequence = revert_interleaving(stem_sequence)
        stem_sequence = cut_sequence(stem_sequence, data_items[i].sequence_lengths.stem[0])
        stem_sequence = remove_special_tokens(stem_sequence, get_special_tokens())
        stem_tensor, sr = decode(stem_sequence, device)
        save_audio(stem_tensor.detach().cpu(), stem_file, sample_rate=sr)

        print(f'Generating {AUDIO_TO_GENERATE_LENGTH} seconds of audio')
        start_time = time.perf_counter()
        output_sequence = complete_sequence(
            model,
            device,
            input_sequence,
            frame_rate=frame_rate,
            max_output_length_in_seconds=AUDIO_TO_GENERATE_LENGTH,
            padding_value=get_pad_token_id(),
            use_cache=False,
            show_progress=True,
            temperature=0.8,
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
    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path(), compile=False)
    data_loader = get_data_loader('train')
    codec = get_codec(device)
    frame_rate = codec.config.frame_rate

    test_model(model, device, data_loader, frame_rate)
    
