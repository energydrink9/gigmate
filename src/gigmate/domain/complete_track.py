from gigmate.utils.device import get_device
from gigmate.model.model import get_model
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.domain.prediction import complete_audio_file


def complete_track(file, model, device):
    complete_audio_file(file, model, device)


if __name__ == '__main__':
    device = get_device()
    model = get_model(device=device, checkpoint_path=get_latest_model_checkpoint_path())    
    complete_track('output/nirvana.ogg', model, device)