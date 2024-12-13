import itertools
import math
import os
import time
from encodec.utils import save_audio
from s3fs.core import S3FileSystem

from gigmate.dataset.dataset import get_data_loader
from gigmate.model.codec import decode, get_codec
from gigmate.model.model import TransformerModel, get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path, get_task
from gigmate.domain.prediction import complete_sequence
from gigmate.utils.constants import MAX_DECODER_SEQ_LEN, get_pad_token_id
from gigmate.utils.device import get_device, Device
from gigmate.utils.sequence_utils import cut_sequence, revert_interleaving

NUM_OUTPUT_FILES = 5
SUBSET_OF_TEST_DATASET_NUMBER = 5
AUDIO_TO_GENERATE_LENGTH = 4
TEMPERATURE = 1.0
BUCKET_NAME = 'gigmate-predictions'
INPUT_SEQUENCE_LENGTH_IN_SECONDS = 2


def test_model(model: TransformerModel, device: Device, data_loader, frame_rate: int, fs: S3FileSystem, id: str = 'default') -> None:
    data_items = list(itertools.islice(iter(data_loader), SUBSET_OF_TEST_DATASET_NUMBER * NUM_OUTPUT_FILES, (SUBSET_OF_TEST_DATASET_NUMBER + 1) * NUM_OUTPUT_FILES))

    for i in list(range(0, NUM_OUTPUT_FILES)):
        file_idx = SUBSET_OF_TEST_DATASET_NUMBER * NUM_OUTPUT_FILES + i

        full_track_file = f'/tmp/full_track_{i}.wav'
        stem_file = f'/tmp/stem_{i}.wav'
        output_file = f'/tmp/output_{i}.wav'
        
        print(f'File: {data_items[i].paths[0]}')
        print(f'Generating audio file output {i}:')

        full_track_sequence = data_items[i].inputs.full_track[:1, :, :]
        sequence_length = data_items[i].sequence_lengths.full_track[0]
        full_track_sequence = cut_sequence(full_track_sequence, sequence_length, cut_left=True)
        full_track_tensor, sr = decode(full_track_sequence, device)
        save_audio(full_track_tensor.detach().cpu(), full_track_file, sample_rate=sr)
        upload_name = f'full_track_{file_idx}.wav'
        fs.upload(full_track_file, os.path.join(BUCKET_NAME, id, upload_name))

        stem_sequence = data_items[i].inputs.stem[:1, :, :]
        stem_sequence = cut_sequence(stem_sequence, data_items[i].sequence_lengths.stem[0], cut_left=False)
        stem_sequence = revert_interleaving(stem_sequence)
        stem_tensor, sr = decode(stem_sequence, device)
        save_audio(stem_tensor.detach().cpu(), stem_file, sample_rate=sr)
        upload_name = f'stem_{file_idx}.wav'
        fs.upload(stem_file, os.path.join(BUCKET_NAME, id, upload_name))

        # Remove start token and cut to desired length
        end = 1 + min(math.ceil(frame_rate * INPUT_SEQUENCE_LENGTH_IN_SECONDS), MAX_DECODER_SEQ_LEN)
        input_sequence = stem_sequence[:, :, 1:end]

        print(f'Generating {AUDIO_TO_GENERATE_LENGTH} seconds of audio')
        start_time = time.perf_counter()
        output_sequence = complete_sequence(
            model,
            device,
            full_track_sequence,
            frame_rate=frame_rate,
            max_output_length_in_seconds=AUDIO_TO_GENERATE_LENGTH,
            padding_value=get_pad_token_id(),
            use_cache=False,
            show_progress=True,
            temperature=TEMPERATURE,
            input_sequence=input_sequence,
        )
        end_time = time.perf_counter()
        print(f'Prediction took {end_time - start_time} seconds')
        output_tensor, sr = decode(output_sequence, device)
        end_time = time.perf_counter()
        print(f'Prediction + decoding took {end_time - start_time} seconds')
        save_audio(output_tensor.detach().cpu(), output_file, sample_rate=sr)
        print(f'Saved predicted file at {output_file}')
        
        upload_name = f'prediction_{file_idx}.wav'
        print(f'copy {output_file} to {os.path.join(BUCKET_NAME, id, upload_name)}')
        fs.upload(output_file, os.path.join(BUCKET_NAME, id, upload_name))
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
    
