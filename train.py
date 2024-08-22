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
from tokenizer import get_tokenizer
from constants import get_params
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

def get_entry(entry, device):
    return entry['input_ids'].to(device), entry['labels'].to(device)

def get_loss_fn(pad_token_id):
    return nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Ignore padding index

accuracy_metric = torchmetrics.classification.Accuracy(task="binary", ignore_index=tokenizer.pad_token_id)
perplexity_metric = Perplexity(ignore_index=tokenizer.pad_token_id)
f1_metric = F1Score(task="binary", num_classes=2, ignore_index=tokenizer.pad_token_id)
precision_metric = Precision(task="binary", average='macro', num_classes=2, ignore_index=tokenizer.pad_token_id)
recall_metric = Recall(task="binary", average='macro', num_classes=2, ignore_index=tokenizer.pad_token_id)

# define the LightningModule
class ModelTraining(L.LightningModule):
    def __init__(self, model, learning_rate, pad_token_id):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.pad_token_id = pad_token_id

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # Assuming optimizer is defined
        return optimizer
    
    def log_metrics(self, dataset, **kwargs):
        for (key, value) in kwargs:
            self.log_metric(dataset, key, value)

    def log_metric(self, dataset, metric_name: str, value):
        self.log(f"{dataset}_{metric_name}", value)

    def compute_metrics(self, predictions, targets):
        accuracy = accuracy_metric(predictions, targets)
        perplexity = perplexity_metric(predictions, targets)
        f1_score = f1_metric(predictions, targets)
        precision = precision_metric(predictions, targets)
        recall = recall_metric(predictions, targets)

        return accuracy, perplexity, f1_score, precision, recall

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.model(inputs)
        outputs_for_loss = predictions.transpose(1, 2)

        loss = get_loss_fn(self.pad_token_id)(outputs_for_loss, targets)
        metrics = self.compute_metrics(predictions, targets)
        self.log_metrics("train", loss, **metrics)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.model(inputs)
        outputs_for_loss = predictions.transpose(1, 2)
        loss = get_loss_fn(self.pad_token_id)(outputs_for_loss, targets)

        metrics = self.compute_metrics(predictions, targets)
        self.log_metrics("val", loss, **metrics)

        return loss

def train_model():
    params = get_params()
    print(f'Running with parameters: {params}')

    device = get_device()    
    print(f'Running on device: {device}')
    #torch.set_default_device(device)

    model_training = ModelTraining(model, learning_rate = params['learning_rate'])

    tokenizer = get_tokenizer()
    model = get_model()
    model.to(device)

    #summary(model, input_size=(BATCH_SIZE, max_seq_len, vocab_size))
    print('Loaded model:')
    print(model)

    train_loader, validation_loader, _ = get_data_loaders()

    logger = TensorBoardLogger("tb_logs", name="GigMate")
    early_stopping = EarlyStopping('val_loss')
    trainer = L.Trainer(callbacks=[early_stopping], logger=logger)
    trainer.fit(model_training, train_loader, validation_loader, tokenizer=tokenizer, max_epochs=params['epochs'])


if __name__ == '__main__':
    train_model()