from gigmate.domain.midi_conversion import convert_audio_to_midi
from gigmate.utils.constants import get_params
from gigmate.utils.device import get_device
from gigmate.model.model import get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.domain.predict import create_midi_from_sequence
from gigmate.domain.prediction import complete_sequence
from gigmate.model.tokenizer import get_tokenizer
from symusic import Score

def complete_midi_track(score: Score, model, device):
    max_seq_len = get_params()['max_seq_len']
    tokenizer = get_tokenizer()
    score = tokenizer(score)
    input_sequence = score.ids
    input_sequence = input_sequence[:4096]
    create_midi_from_sequence(tokenizer, input_sequence, 'output/input_nirvana.mid')
    output_sequence = complete_sequence(model, device, tokenizer, input_sequence, max_seq_len=max_seq_len, max_output_tokens=300, max_output_length_in_seconds=10, show_progress=True)
    create_midi_from_sequence(tokenizer, output_sequence, 'output/output_nirvana.mid')

def complete_track(file, model, device):
    score = convert_audio_to_midi(file)
    complete_midi_track(score, model, device)

if __name__ == '__main__':
    device = get_device()
    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())    
    complete_track('output/nirvana_cut.mp3', model, device)