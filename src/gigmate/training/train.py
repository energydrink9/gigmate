import os
from clearml import Task
import torch
from gigmate.dataset.dataset import get_data_loaders
from gigmate.model.model_checkpoint import get_latest_model_checkpoint_path
from gigmate.training.training_model import get_training_model
from gigmate.utils.constants import get_clearml_project_name, get_params, get_random_seed
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from gigmate.utils.device import get_device

DEBUG = False
OUTPUT_DIRECTORY = 'output'
UPLOAD_CHECKPOINT = True
UPLOAD_CHECKPOINT_EVERY_N_EPOCHS = 3
MIXED_PRECISION = True

L.seed_everything(get_random_seed())


def upload_weights(task, epoch, filepath):
    if UPLOAD_CHECKPOINT:
        task.upload_artifact(name=f'weights-epoch-{epoch}', artifact_object=filepath, wait_on_upload=False)


def init_clearml_task(params):
    
    task_name = f'train enc_seq_len {params["max_seq_len"]}'
    task_name += f' dec_seq_len {params["max_decoder_seq_len"]}'
    task_name += f' batch {params["batch_size"]}'
    task_name += f' layers {params["encoder_layers"]}-{params["decoder_layers"]}'
    task_name += f' heads {params["num_heads"]}'
    task_name += f' lr {params["learning_rate"]}'
    task_name += f' dff {params["dff"]}'
    
    task = Task.init(
        project_name=get_clearml_project_name(),
        task_name=task_name,
    )

    task.connect(params)

    # Workaround to prevent pytorch.compile errors due to builtins patching performed by clearml
    import builtins
    builtins.__import__ = builtins.__org_import__

    return task


class ModelCheckpointUpload(ModelCheckpoint):
    def __init__(self, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
    
    def _save_checkpoint(self, trainer, filepath) -> None:
        super()._save_checkpoint(trainer, filepath)
        upload_weights(self.task, trainer.current_epoch, filepath)


def train_model(task, params, device, output_dir, train_loader, validation_loader, ckpt_path=None):
    accumulate_grad_batches = params['accumulate_grad_batches']
    steps_per_epoch = len(train_loader) // accumulate_grad_batches
    training_model, quantizer = get_training_model(params, ckpt_path, device, task, steps_per_epoch)

    # summary(model, input_size=(params['batch_size'], max_seq_len, vocab_size))
    print('Loaded model:')
    print(training_model.model)

    logger = TensorBoardLogger("tb_logs", name="GigMate")
    # early_stopping = EarlyStopping('val_loss', patience=10)
    checkpoint_callback = ModelCheckpointUpload(
        task=task,
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='{epoch}',
        every_n_epochs=UPLOAD_CHECKPOINT_EVERY_N_EPOCHS,
        enable_version_counter=True,
        save_top_k=1,
        save_weights_only=True,
        save_on_train_epoch_end=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = L.Trainer(
        # callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        max_epochs=params['epochs'],
        limit_train_batches=params['training_set_size'],
        limit_val_batches=params['validation_set_size'],
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=params['gradient_clip'],
        precision='16-mixed' if device == 'cuda' and MIXED_PRECISION is True else '32-true',
        detect_anomaly=DEBUG,
        deterministic='warn',
        check_val_every_n_epoch=1,
    )
    
    trainer.fit(training_model, train_loader, validation_loader)

    model = trainer.lightning_module
    return quantizer.convert(model) if quantizer is not None else model


if __name__ == '__main__':
    device = get_device()
    print(f'Running torch version: {torch.__version__}')
    print(f'Running on device: {device}')

    params = get_params()
    print(f'Running with parameters: {params}')

    task = init_clearml_task(params)

    print('Loading dataset...')
    train_loader, validation_loader, _ = get_data_loaders()

    # Set to none to start training from scratch, otherwise use `get_latest_model_checkpoint_path` to continue training from last checkpoint.
    ckpt_path = get_latest_model_checkpoint_path()
    model = train_model(task, params, device, OUTPUT_DIRECTORY, train_loader, validation_loader, ckpt_path=ckpt_path)
