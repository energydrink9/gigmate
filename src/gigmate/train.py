from typing import Literal
from clearml import Task
import torchmetrics
import torch
from torch import nn
from torchmetrics.text import Perplexity
from torchmetrics import F1Score, Precision, Recall
from gigmate.dataset import get_data_loaders
from gigmate.model import get_model
import numpy as np
from gigmate.constants import get_clearml_project_name, get_params, get_pad_token_id, get_random_seed
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from gigmate.device import get_device
from gigmate.predict import test_model

WEIGHTS_FILE = 'output/gigmate.weights'
LOG_INTERVAL = 5

pad_token_id = get_pad_token_id()
batch_size = get_params()['batch_size']
L.seed_everything(get_random_seed())
#torch.use_deterministic_algorithms(True) # TODO: re-enable

def init_clearml_task(params):
    task = Task.init(
            project_name=get_clearml_project_name(),
            task_name=f'train seq {params["max_seq_len"]} batch {params["batch_size"]} layers {params["num_layers"]} heads {params["num_heads"]} lr {params["learning_rate"]} dff {params["dff"]}',
    )

    task.connect(params)

    return task

def get_inputs_and_targets(batch, device):
    return batch['input_ids'].to(device), batch['labels'].to(device)

class ModelTraining(L.LightningModule):
    def __init__(self, model, learning_rate, vocab_size):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.train_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Ignore padding index
        self.val_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Ignore padding index

        self.train_accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)
        self.val_accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)

        # self.train_perplexity_metric = Perplexity(ignore_index=pad_token_id)
        # self.val_perplexity_metric = Perplexity(ignore_index=pad_token_id)

        self.train_f1_metric = F1Score(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)
        self.val_f1_metric = F1Score(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)

        self.train_precision_metric = Precision(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)
        self.val_precision_metric = Precision(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)

        self.train_recall_metric = Recall(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)
        self.val_recall_metric = Recall(task="multiclass", num_classes=vocab_size, ignore_index=pad_token_id)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def log_metrics(self, dataset, **kwargs):
        for (key, value) in kwargs.items():
            self.log_metric(dataset, key, value)

    def log_metric(self, dataset: Literal['train', 'val', 'test'], metric_name: str, value):
        log_on_step = dataset == 'train'
        self.log(f"{dataset}_{metric_name}", value, on_step=log_on_step, on_epoch=True, prog_bar=metric_name in ['loss', 'accuracy'])

    def compute_train_loss(self, transposed_logits, targets):
        return self.train_loss(transposed_logits, targets)
    
    def compute_train_metrics(self, transposed_logits, targets):
        accuracy = self.train_accuracy_metric(transposed_logits, targets)
        f1_score = self.train_f1_metric(transposed_logits, targets)
        precision = self.train_precision_metric(transposed_logits, targets)
        recall = self.train_recall_metric(transposed_logits, targets)

        return dict({
            'accuracy': accuracy,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
        })

    def compute_val_loss(self, transposed_logits, targets):
        return self.val_loss(transposed_logits, targets)
    
    def compute_val_metrics(self, transposed_logits, targets):
        accuracy = self.val_accuracy_metric(transposed_logits, targets)
        f1_score = self.val_f1_metric(transposed_logits, targets)
        precision = self.val_precision_metric(transposed_logits, targets)
        recall = self.val_recall_metric(transposed_logits, targets)

        return dict({
            'accuracy': accuracy,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
        })

    def training_step(self, batch, batch_idx):
        inputs, targets = get_inputs_and_targets(batch, self.device)
        logits = self.model(inputs)
        transposed_logits = logits.transpose(1, 2)
        #transposed_logits = logits.permute(0, 2, 1)
        loss = self.compute_train_loss(transposed_logits, targets)
        
        if batch_idx % LOG_INTERVAL == 0:
            metrics = self.compute_train_metrics(transposed_logits, targets)
            self.log_metrics("train", loss=loss, **metrics)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = get_inputs_and_targets(batch, self.device)
        logits = self.model(inputs)
        transposed_logits = logits.transpose(1, 2)
        loss = self.compute_val_loss(transposed_logits, targets)

        if batch_idx % LOG_INTERVAL == 0:
            metrics = self.compute_val_metrics(transposed_logits, targets)
            self.log_metrics("val", loss=loss, **metrics)

        return loss

def train_model(params, device, output_dir, train_loader, validation_loader):

    print('Loading model...')
    model = get_model()
    model.to(device)

    if device == 'cuda':
        model = torch.compile(model)

    model_training = ModelTraining(model, learning_rate = params['learning_rate'], vocab_size=params['vocab_size'])

    #summary(model, input_size=(BATCH_SIZE, max_seq_len, vocab_size))
    print('Loaded model:')
    print(model)

    logger = TensorBoardLogger("tb_logs", name="GigMate")
    early_stopping = EarlyStopping('val_loss')
    
    trainer = L.Trainer(
        callbacks=[early_stopping],
        logger=logger,
        max_epochs=params['epochs'],
        limit_train_batches=params['training_set_size'],
        limit_val_batches=params['validation_set_size'],
        accumulate_grad_batches=params['accumulate_grad_batches'],
    )
    
    trainer.fit(model_training, train_loader, validation_loader)
    trainer.save_checkpoint(output_dir)

    return model


if __name__ == '__main__':
    device = get_device()
    print(f'Running on device: {device}')

    params = get_params()
    print(f'Running with parameters: {params}')

    task = init_clearml_task(params)

    print('Loading dataset...')
    train_loader, validation_loader, _ = get_data_loaders()

    model = train_model(params, device, WEIGHTS_FILE, train_loader, validation_loader)
    task.upload_artifact(name='weights', artifact_object=WEIGHTS_FILE)

    output_midis = test_model(model, device, validation_loader)

    for midi in output_midis:
        task.upload_artifact(name=midi['name'], artifact_object=midi['file'])