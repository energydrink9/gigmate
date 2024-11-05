from gigmate.utils.device import get_device
from gigmate.model.model import get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.domain.prediction import complete_sequence
from pydub import AudioSegment


def complete_track(file, model, device):
    audio_data = AudioSegment.from_file(file)
    score = convert_audio_to_midi(audio_data, device, separator)
    complete_midi_track(score, model, device)


if __name__ == '__main__':
    device = get_device()
    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())    
    complete_track('output/nirvana.ogg', model, device)