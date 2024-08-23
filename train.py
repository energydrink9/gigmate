from typing import Literal
from clearml import Task
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import torch
from torch import nn
from torchmetrics.text import Perplexity
from torchmetrics import F1Score, Precision, Recall
from dataset import get_data_loaders
from model import get_model
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from constants import get_clearml_project_name, get_params, get_pad_token_id
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from device import get_device

WEIGHTS_FILE = 'gigmate.weights'

pad_token_id = get_pad_token_id()

def init_clearml_task(params):
    task = Task.init(
        project_name=get_clearml_project_name(),
        task_name='training',
    )

    task.connect(params)

def get_inputs_and_targets(batch, device):
    return batch['input_ids'].to(device), batch['labels'].to(device)

def get_loss_fn(pad_token_id):
    return nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Ignore padding index

# define the LightningModule
class ModelTraining(L.LightningModule):
    def __init__(self, model, learning_rate, vocab_size):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.train_accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)
        self.val_accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)

        self.train_perplexity_metric = Perplexity(ignore_index=pad_token_id)
        self.val_perplexity_metric = Perplexity(ignore_index=pad_token_id)

        self.train_f1_metric = F1Score(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)
        self.val_f1_metric = F1Score(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)

        self.train_precision_metric = Precision(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)
        self.val_precision_metric = Precision(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)

        self.train_recall_metric = Recall(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)
        self.val_recall_metric = Recall(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # Assuming optimizer is defined
        return optimizer
    
    def log_metrics(self, dataset, **kwargs):
        for (key, value) in kwargs.items():
            self.log_metric(dataset, key, value)

    def log_metric(self, dataset: Literal['train', 'val', 'test'], metric_name: str, value):
        log_on_epoch = dataset == 'val'
        self.log(f"{dataset}_{metric_name}", value, on_step=True, on_epoch=log_on_epoch)

    def compute_train_metrics(self, logits, targets):
        transposed_logits = logits.transpose(1, 2)
        loss = get_loss_fn(pad_token_id)(transposed_logits, targets)
        accuracy = self.train_accuracy_metric(transposed_logits, targets)
        perplexity = self.train_perplexity_metric(logits, targets)
        f1_score = self.train_f1_metric(transposed_logits, targets)
        precision = self.train_precision_metric(transposed_logits, targets)
        recall = self.train_recall_metric(transposed_logits, targets)

        return loss, dict({
            'accuracy': accuracy,
            'perplexity': perplexity,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
        })
    
    def compute_val_metrics(self, logits, targets):
        transposed_logits = logits.transpose(1, 2)
        loss = get_loss_fn(pad_token_id)(transposed_logits, targets)
        accuracy = self.val_accuracy_metric(transposed_logits, targets)
        perplexity = self.val_perplexity_metric(logits, targets)
        f1_score = self.val_f1_metric(transposed_logits, targets)
        precision = self.val_precision_metric(transposed_logits, targets)
        recall = self.val_recall_metric(transposed_logits, targets)

        return loss, dict({
            'accuracy': accuracy,
            'perplexity': perplexity,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
        })

    def training_step(self, batch, batch_idx):
        inputs, targets = get_inputs_and_targets(batch, self.device)
        logits = self.model(inputs)
        loss, metrics = self.compute_train_metrics(logits, targets)
        self.log_metrics("train", loss=loss, **metrics)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = get_inputs_and_targets(batch, self.device)
        logits = self.model(inputs)
        loss, metrics = self.compute_val_metrics(logits, targets)
        self.log_metrics("val", loss=loss, **metrics)

        return loss

def train_model():
    params = get_params()
    print(f'Running with parameters: {params}')

    init_clearml_task(params)

    device = get_device()    
    print(f'Running on device: {device}')
    #torch.set_default_device(device)

    print('Loading dataset...')
    train_loader, validation_loader, _ = get_data_loaders()

    print('Loading model...')
    model = get_model()
    model.to(device)

    model_training = ModelTraining(model, learning_rate = params['learning_rate'], pad_token_id=pad_token_id, vocab_size=params['vocab_size'])

    #summary(model, input_size=(BATCH_SIZE, max_seq_len, vocab_size))
    print('Loaded model:')
    print(model)

    logger = TensorBoardLogger("tb_logs", name="GigMate")
    early_stopping = EarlyStopping('val_loss')
    trainer = L.Trainer(callbacks=[early_stopping], logger=logger, max_epochs=params['epochs'])
    trainer.fit(model_training, train_loader, validation_loader)
    trainer.save_checkpoint(WEIGHTS_FILE)

    return model


if __name__ == '__main__':
    train_model()