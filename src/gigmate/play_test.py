from gigmate.utils.constants import get_pad_token_id
from gigmate.model.model import get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.domain.prediction import complete_audio_file
from gigmate.utils.device import get_device
from pydub import AudioSegment
import pydub.utils

SAMPLE_RATE = 22050
OUTPUT_SAMPLE_RATE = 44100
device = get_device()
model = get_model(checkpoint_path=get_latest_model_checkpoint_path(), device=device)
TEST_GENERATION_FILE = 'resources/test_creep_cut.ogg'
TEST_TIMING_FILE = 'resources/test_timing_2.wav'
MIDI_PROGRAM = None


# TODO: fix and re-enable
def skip_test_generation_of_audio():
    conditioning_file = TEST_GENERATION_FILE
    output_filename = 'output/completed_audio.ogg'
    conditioning_segment = AudioSegment.from_file(conditioning_file)

    completion_segment = complete_audio_file(model, device, conditioning_file, max_output_length_in_seconds=10, padding_value=get_pad_token_id())
    completed_segment = conditioning_segment + completion_segment
    completed_segment.export(output_filename, format='ogg', codec="opus")

    print(f'Generated completed audio file at: {output_filename}')


# TODO: fix and re-enable
def skip_test_timing_of_generated_audio_with_processing_time():
    conditioning_file = TEST_GENERATION_FILE
    output_filename = 'output/completed_audio.ogg'
    seconds_passed = 2.8

    conditioning_file_info = pydub.utils.mediainfo(conditioning_file)
    frames_to_remove = int(conditioning_file_info['sample_rate']) * seconds_passed

    conditioning_segment = AudioSegment.from_file(conditioning_file)
    conditioning_segment_cut = conditioning_segment[:-frames_to_remove]

    completion_segment = complete_audio_file(model, device, conditioning_segment_cut, max_output_length_in_seconds=10, padding_value=get_pad_token_id())
    completed_segment = conditioning_segment + completion_segment[frames_to_remove:]
    completed_segment.export(output_filename, format='ogg', codec="opus")

    print(f'Generated completed audio file at: {output_filename}')