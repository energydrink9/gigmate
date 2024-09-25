from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
from neural_compressor.quantization import fit

from gigmate.dataset.dataset import get_data_loader
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.training.training_model import get_training_model
from gigmate.utils.constants import get_params
from gigmate.utils.device import get_device

def quantize():
    device = get_device()
    training_model = get_training_model(params = get_params(), checkpoint_path=get_latest_model_checkpoint_path(), device=device)
    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
    tuning_criterion = TuningCriterion(max_trials=600)
    conf = PostTrainingQuantConfig(
        approach="static", backend="default", tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion
    )
    dataloader = get_data_loader('validation')

    q_model = fit(model=training_model.model, conf=conf, calib_dataloader=dataloader)
    q_model.save('output')

if __name__ == '__main__':
    quantize()