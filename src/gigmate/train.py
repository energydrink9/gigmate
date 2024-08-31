from typing import Literal
from clearml import Task
import torchmetrics
import torch
from torch import nn
from torchmetrics.text import Perplexity
from torch.optim.lr_scheduler import CyclicLR
from gigmate.dataset import get_data_loaders
from gigmate.model import get_model
from gigmate.constants import get_clearml_project_name, get_params, get_pad_token_id, get_random_seed
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from gigmate.device import get_device
from gigmate.predict import test_model
import os

OUTPUT_DIRECTORY = 'output'
LOG_INTERVAL = 5
LOAD_TASK_WEIGHTS = None#'train seq 256 batch 128 layers 12 heads 8 lr 0.0001 dff 1024'
UPLOAD_WEIGHTS = True

pad_token_id = get_pad_token_id()
batch_size = get_params()['batch_size']
L.seed_everything(get_random_seed())
#torch.use_deterministic_algorithms(True) TODO: Enable this

def get_checkpoint_dir(output_dir):
    return os.path.join(output_dir, 'checkpoints')

def get_weights_path(output_dir):
    return os.path.join(get_checkpoint_dir(output_dir), f'epoch={epoch}.ckpt')

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

def get_task_artifact(task_name: str, artifact_name: str):
    preprocess_task = Task.get_task(task_name=task_name, project_name=get_clearml_project_name())
    return preprocess_task.artifacts[artifact_name].get_local_copy()

def get_inputs_and_targets(batch, device):
    return batch['input_ids'].to(device), batch['labels'].to(device)

def load_ckpt(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location=torch.device(device), weights_only=True)['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace("model._orig_mod.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)

class ModelCheckpointUpload(ModelCheckpoint):
    def __init__(self, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
    
    def _save_checkpoint(self, trainer, filepath) -> None:
        super()._save_checkpoint(trainer, filepath)
        upload_weights(self.task, trainer.current_epoch, filepath)


class ModelTraining(L.LightningModule):
    def __init__(self, model, learning_rate, max_learning_rate, step_size_up, vocab_size):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.step_size_up = step_size_up
        self.max_learning_rate = max_learning_rate

        self.train_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Ignore padding index
        self.val_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Ignore padding index

        self.train_accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)
        self.val_accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)

        self.train_perplexity_metric = Perplexity(ignore_index=pad_token_id)
        self.val_perplexity_metric = Perplexity(ignore_index=pad_token_id)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        step_size_up = self.step_size_up

        scheduler = CyclicLR(
            optimizer,
            base_lr=self.learning_rate,
            max_lr=self.max_learning_rate,
            step_size_up=step_size_up,
            mode='triangular2',
            cycle_momentum=False
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def log_metrics(self, dataset, **kwargs):
        for (key, value) in kwargs.items():
            self.log_metric(dataset, key, value)

    def log_metric(self, dataset: Literal['train', 'val', 'test'], metric_name: str, value):
        log_on_step = dataset == 'train'
        self.log(f"{dataset}_{metric_name}", value, on_step=log_on_step, on_epoch=True, prog_bar=metric_name in ['loss', 'accuracy', 'perplexity'])

    def compute_train_loss(self, transposed_logits, targets):
        return self.train_loss(transposed_logits, targets)
    
    def compute_train_metrics(self, logits, transposed_logits, targets):
        accuracy = self.train_accuracy_metric(transposed_logits, targets)
        perplexity = self.train_perplexity_metric(logits, targets)

        return dict({
            'accuracy': accuracy,
            'perplexity': perplexity,
        })

    def compute_val_loss(self, transposed_logits, targets):
        return self.val_loss(transposed_logits, targets)
    
    def compute_val_metrics(self, logits, transposed_logits, targets):
        accuracy = self.val_accuracy_metric(transposed_logits, targets)
        perplexity = self.val_perplexity_metric(logits, targets)

        return dict({
            'accuracy': accuracy,
            'perplexity': perplexity,
        })

    def training_step(self, batch, batch_idx):
        inputs, targets = get_inputs_and_targets(batch, self.device)
        logits = self.model(inputs)
        transposed_logits = logits.transpose(1, 2) # Also equivalent to: transposed_logits = logits.permute(0, 2, 1)
        loss = self.compute_train_loss(transposed_logits, targets)

        if batch_idx % LOG_INTERVAL == 0:
            metrics = self.compute_train_metrics(logits, transposed_logits, targets)
            self.log_metrics("train", loss=loss, **metrics)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = get_inputs_and_targets(batch, self.device)
        logits = self.model(inputs)
        transposed_logits = logits.transpose(1, 2)
        loss = self.compute_val_loss(transposed_logits, targets)

        if batch_idx % LOG_INTERVAL == 0:
            metrics = self.compute_val_metrics(logits, transposed_logits, targets)
            self.log_metrics("val", loss=loss, **metrics)

        return loss

def train_model(task, params, device, output_dir, train_loader, validation_loader, ckpt_path = None):

    print('Loading model...')
    model = get_model(params)

    if ckpt_path is not None:
        load_ckpt(model, ckpt_path)

    model.to(device)

    if device == 'cuda':
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    model_training = ModelTraining(model, learning_rate = params['learning_rate'], step_size_up=params['step_size_up'], max_learning_rate=params['max_learning_rate'], vocab_size=params['vocab_size'])

    #summary(model, input_size=(BATCH_SIZE, max_seq_len, vocab_size))
    print('Loaded model:')
    print(model)

    logger = TensorBoardLogger("tb_logs", name="GigMate")
    early_stopping = EarlyStopping('val_loss')
    checkpoint_callback = ModelCheckpointUpload(
        task=task,
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='{epoch}',
        every_n_epochs=1,
        enable_version_counter=True,
        save_top_k=1,
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
    )
    
    trainer.fit(model_training, train_loader, validation_loader)
    weights_file = get_weights_path(output_dir)
    trainer.save_checkpoint(weights_file)

    return model


if __name__ == '__main__':
    device = get_device()
    print(f'Running on device: {device}')

    params = get_params()
    print(f'Running with parameters: {params}')

    model = get_model(params)    
    task = init_clearml_task(params)

    print('Loading dataset...')
    train_loader, validation_loader, _ = get_data_loaders()

    ckpt_path = get_task_artifact(LOAD_TASK_WEIGHTS, 'weights') if LOAD_TASK_WEIGHTS is not None else None

    model = train_model(task, params, device, OUTPUT_DIRECTORY, train_loader, validation_loader, ckpt_path=ckpt_path)

    output_midis = test_model(model, device, validation_loader)

    for midi in output_midis:
        task.upload_artifact(name=midi['name'], artifact_object=midi['file'])