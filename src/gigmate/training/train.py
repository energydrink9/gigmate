import os
import shutil
from typing import Optional, cast
from clearml import Task
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from s3fs.core import S3FileSystem

from gigmate.dataset.dataset import get_data_loaders
from gigmate.model.model_checkpoint import CHECKPOINTS_DIR, LATEST_CHECKPOINT_EPOCH_PATH, LATEST_CHECKPOINT_PATH, get_latest_model_checkpoint_path, get_remote_checkpoint_path
from gigmate.training.training_model import get_training_model
from gigmate.utils.constants import get_clearml_project_name, get_params
from gigmate.utils.device import get_device
from gigmate.utils.env import get_environment

ENVIRONMENT = get_environment()
DEBUG = False
UPLOAD_CHECKPOINT = True
UPLOAD_CHECKPOINT_EVERY_N_EPOCHS = 4
MIXED_PRECISION = True
USE_CLEARML = ENVIRONMENT != 'dev'
VAL_CHECK_INTERVAL = None


def upload_weights(fs: S3FileSystem, task_id: str, epoch: int, filepath: str):
    if UPLOAD_CHECKPOINT:
        remote_path = get_remote_checkpoint_path(task_id, epoch)
        fs.upload(filepath, remote_path)


def init_clearml_task(params):
    
    task_name = f'train dmodel {params["d_model"]},'
    task_name += f' batch {params["batch_size"]},'
    task_name += f' layers {params["encoder_layers"]}/{params["decoder_layers"]},'
    task_name += f' heads {params["num_heads"]},'
    task_name += f' seq_len {params["max_seq_len"]}/{params["max_decoder_seq_len"]}'
    
    task = Task.init(
        project_name=get_clearml_project_name(),
        task_name=task_name,
    )

    task.connect(params)

    # Workaround to prevent pytorch.compile errors due to builtins patching performed by clearml
    import builtins
    builtins.__import__ = builtins.__org_import__  # type: ignore

    return task


class ModelCheckpointUpload(ModelCheckpoint):
    def __init__(self, task_id: str, fs: S3FileSystem, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_id = task_id
        self.fs = fs
    
    def _save_checkpoint(self, trainer, filepath) -> None:
        super()._save_checkpoint(trainer, filepath)
        shutil.copyfile(filepath, LATEST_CHECKPOINT_PATH)
        with open(LATEST_CHECKPOINT_PATH, 'w') as f:
            f.write('{}'.format(trainer.current_epoch))
        if trainer.current_epoch % UPLOAD_CHECKPOINT_EVERY_N_EPOCHS == 0:
            upload_weights(self.fs, self.task_id, trainer.current_epoch, filepath)


def train_model(task: Optional[Task], params, device, train_loader: DataLoader, validation_loader: DataLoader, fs: Optional[S3FileSystem], ckpt_path: Optional[str] = None, resume_epoch: Optional[int] = None):
    accumulate_grad_batches = params['accumulate_grad_batches']
    steps_per_epoch = len(train_loader) // accumulate_grad_batches
    training_model = get_training_model(params, ckpt_path, device, task, steps_per_epoch, resume_epoch=resume_epoch, compile=False)

    # summary(model, input_size=(params['batch_size'], max_seq_len, vocab_size))
    print('Loaded model:')
    print(training_model.model)

    logger = TensorBoardLogger("tb_logs", name="GigMate")
    # early_stopping = EarlyStopping('val_loss', patience=10)
    checkpoint_callback = ModelCheckpointUpload(
        task_id=cast(str, task.id) if task is not None else 'default',
        fs=fs,
        dirpath=CHECKPOINTS_DIR,
        filename='{epoch}',
        every_n_epochs=1,
        enable_version_counter=True,
        save_top_k=1,
        save_weights_only=True,
        save_on_train_epoch_end=True,
    ) if fs is not None else None
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = L.Trainer(
        # callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        callbacks=[checkpoint_callback, lr_monitor] if checkpoint_callback is not None else [lr_monitor],
        logger=logger,
        max_epochs=params['epochs'],
        limit_train_batches=params['training_set_size'],
        limit_val_batches=params['validation_set_size'],
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=params['gradient_clip'],
        precision='16-mixed' if device == 'cuda' and MIXED_PRECISION is True else '32-true',
        detect_anomaly=DEBUG,
        # deterministic='warn',
        check_val_every_n_epoch=1,
        val_check_interval=VAL_CHECK_INTERVAL,
    )
    
    trainer.fit(training_model, train_loader, validation_loader)

    model = trainer.lightning_module
    return model


if __name__ == '__main__':
    device = get_device()
    print(f'Running torch version: {torch.__version__}')
    print(f'Running on device: {device}')

    params = get_params()
    print(f'Running with hyper-parameters: {params}')

    task = None

    if USE_CLEARML is True:
        task = init_clearml_task(params)

    fs = S3FileSystem(use_listings_cache=False) if UPLOAD_CHECKPOINT is True else None

    print('Loading dataset...')
    train_loader, validation_loader, _ = get_data_loaders()

    ckpt_path = get_latest_model_checkpoint_path()

    if os.path.exists(LATEST_CHECKPOINT_EPOCH_PATH):
        with open(LATEST_CHECKPOINT_EPOCH_PATH, 'r') as f:
            resume_epoch: Optional[int] = int(f.readline())

        print(f'Resuming training from epoch {resume_epoch}')

    else:
        resume_epoch = None

    model = train_model(task, params, device, train_loader, validation_loader, fs=fs, ckpt_path=ckpt_path, resume_epoch=resume_epoch)
