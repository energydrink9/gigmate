import itertools
import os
import time
from encodec.utils import save_audio
import torch
from s3fs.core import S3FileSystem

from gigmate.dataset.dataset import get_data_loader
from gigmate.model.codec import decode, get_codec
from gigmate.model.model import TransformerModel, get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path, get_task
from gigmate.domain.prediction import complete_sequence
from gigmate.utils.constants import get_pad_token_id
from gigmate.utils.device import get_device, Device
from gigmate.utils.sequence_utils import cut_sequence, revert_interleaving

NUM_OUTPUT_FILES = 5
SUBSET_OF_TEST_DATASET_NUMBER = 0
AUDIO_TO_GENERATE_LENGTH = 10
TEMPERATURE = 0
# NUMBER_OF_SECONDS_FOR_PREDICTION = 10
BUCKET_NAME = 'gigmate-predictions'
PROTOCOL = 's3://'
BUCKET = f'{PROTOCOL}{BUCKET_NAME}'


def test_model(model: TransformerModel, device: Device, data_loader, frame_rate: int, fs: S3FileSystem, id: str = 'default') -> None:
    data_items = list(itertools.islice(iter(data_loader), SUBSET_OF_TEST_DATASET_NUMBER * NUM_OUTPUT_FILES, (SUBSET_OF_TEST_DATASET_NUMBER + 1) * NUM_OUTPUT_FILES))

    for i in list(range(0, NUM_OUTPUT_FILES)):
        input_file = f'/tmp/input_{i}.wav'
        stem_file = f'/tmp/stem_{i}.wav'
        output_file = f'/tmp/output_{i}.wav'
        
        print(f'Generating audio file output {i}:')
        # input_file = 'resources/test_generation.wav'

        torch.set_printoptions(edgeitems=20)
        print('Input sequence', data_items[i].inputs.full_track[:1, :, :].shape, data_items[i].inputs.full_track[:1, :, :])
        print('Expected Stem', data_items[i].inputs.stem[:1, :, :].shape, data_items[i].inputs.stem[:1, :, :])

        input_sequence = data_items[i].inputs.full_track[:1, :, :]
        input_sequence = cut_sequence(input_sequence, data_items[i].sequence_lengths.full_track[0], cut_left=True)
        input_sequence = revert_interleaving(input_sequence)
        input_tensor, sr = decode(input_sequence, device)
        save_audio(input_tensor.detach().cpu(), input_file, sample_rate=sr)

        stem_sequence = data_items[i].inputs.stem[:1, :, :]
        stem_sequence = cut_sequence(stem_sequence, data_items[i].sequence_lengths.stem[0], cut_left=False)
        stem_sequence = revert_interleaving(stem_sequence)
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
            temperature=TEMPERATURE,
        )
        end_time = time.perf_counter()
        print(f'Prediction took {end_time - start_time} seconds')
        output_tensor, sr = decode(output_sequence, device)
        end_time = time.perf_counter()
        print(f'Prediction + decoding took {end_time - start_time} seconds')
        save_audio(output_tensor.detach().cpu(), output_file, sample_rate=sr)
        print(f'Saved predicted file at {output_file}')
        
        file_idx = SUBSET_OF_TEST_DATASET_NUMBER * NUM_OUTPUT_FILES + i
        upload_name = f'prediction_{file_idx}'
        fs.copy(output_file, os.path.join(BUCKET, id, upload_name))
        print(f'Uploaded file with artifact name {upload_name}')


if __name__ == '__main__':
    fs = S3FileSystem(use_listings_cache=False)
    device = get_device()
    task = get_task()
    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path(), compile=False)
    data_loader = get_data_loader('validation')
    codec = get_codec(device)
    frame_rate = codec.config.frame_rate

    test_model(model, device, data_loader, frame_rate, fs=fs, id=task.id if task is not None and task.id is not None else 'default')
    
