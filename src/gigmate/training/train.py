from clearml import Task
from gigmate.dataset.dataset import get_data_loaders
from gigmate.training.training_model import get_training_model
from gigmate.utils.constants import get_clearml_project_name, get_params, get_random_seed
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from gigmate.utils.device import get_device
from gigmate.domain.predict import test_model
import os

OUTPUT_DIRECTORY = 'output'
UPLOAD_WEIGHTS = True

BATCH_SIZE = get_params()['batch_size']
L.seed_everything(get_random_seed())
#torch.use_deterministic_algorithms(True) TODO: Enable this

def upload_weights(task, epoch, filepath):
    if UPLOAD_WEIGHTS:
        task.upload_artifact(name=f'weights-epoch-{epoch}', artifact_object=filepath, wait_on_upload=False)

def init_clearml_task(params):
    task = Task.init(
            project_name=get_clearml_project_name(),
            task_name=f'train seq {params["max_seq_len"]} batch {params["batch_size"]} layers {params["num_layers"]} heads {params["num_heads"]} lr {params["learning_rate"]} dff {params["dff"]}',
    )

    task.connect(params)

    return task

class ModelCheckpointUpload(ModelCheckpoint):
    def __init__(self, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
    
    def _save_checkpoint(self, trainer, filepath) -> None:
        super()._save_checkpoint(trainer, filepath)
        upload_weights(self.task, trainer.current_epoch, filepath)

def train_model(task, params, device, output_dir, train_loader, validation_loader, ckpt_path = None):

    training_model = get_training_model(params, ckpt_path, device)

    #summary(model, input_size=(BATCH_SIZE, max_seq_len, vocab_size))
    print('Loaded model:')
    print(training_model.model)

    logger = TensorBoardLogger("tb_logs", name="GigMate")
    early_stopping = EarlyStopping('val_loss')
    checkpoint_callback = ModelCheckpointUpload(
        task=task,
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='{epoch}',
        every_n_epochs=1,
        enable_version_counter=True,
        save_top_k=1,
        save_weights_only=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = L.Trainer(
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        logger=logger,
        max_epochs=params['epochs'],
        limit_train_batches=params['training_set_size'],
        limit_val_batches=params['validation_set_size'],
        accumulate_grad_batches=params['accumulate_grad_batches'],
        gradient_clip_val=params['gradient_clip'],
        precision=16 if device == 'cuda' else 32,
    )
    
    trainer.fit(training_model, train_loader, validation_loader)

    return training_model


if __name__ == '__main__':
    device = get_device()
    print(f'Running on device: {device}')

    params = get_params()
    print(f'Running with parameters: {params}')

    task = init_clearml_task(params)

    print('Loading dataset...')
    train_loader, validation_loader, _ = get_data_loaders()

    training_model = train_model(task, params, device, OUTPUT_DIRECTORY, train_loader, validation_loader, ckpt_path=None)

    output_midis = test_model(training_model.model, device, validation_loader)

    for midi in output_midis:
        task.upload_artifact(name=midi['name'], artifact_object=midi['file'])