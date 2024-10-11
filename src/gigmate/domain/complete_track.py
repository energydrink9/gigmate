from gigmate.utils.device import get_device
from gigmate.model.model import get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.domain.prediction import complete_sequence
from pydub import AudioSegment


def complete_midi_track(score: Score, model, device):
    tokenizer = get_tokenizer()
    score = tokenizer(score)
    input_sequence = score.ids
    input_sequence = input_sequence[:4096]
    create_midi_from_sequence(tokenizer, input_sequence, 'output/input_creep.mid')
    output_sequence = complete_sequence(model, device, tokenizer, input_sequence, max_output_tokens=300, max_output_length_in_seconds=10, padding_value=get_pad_token_id(), show_progress=True)
    create_midi_from_sequence(tokenizer, output_sequence, 'output/output_creep.mid')


def complete_track(file, model, device):
    audio_data = AudioSegment.from_file(file)
    score = convert_audio_to_midi(audio_data, device, separator)
    complete_midi_track(score, model, device)


if __name__ == '__main__':
    device = get_device()
    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())    
    complete_track('output/nirvana.ogg', model, device)