from typing import Optional, Tuple
import torchmetrics.classification
from torch import nn
import lightning as L
from torchmetrics.text import Perplexity
from torch.optim.lr_scheduler import CyclicLR
from typing import Literal
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from gigmate.model.model import get_model
from gigmate.utils.constants import get_pad_token_id
from torchao.quantization.prototype.qat.api import Int8DynActInt4WeightQATQuantizer

LOG_INTERVAL = 5
PAD_TOKEN_ID = get_pad_token_id()

def get_inputs_and_targets(batch, device):
    return batch['input_ids'].to(device), batch['labels'].to(device)

class TrainingModel(L.LightningModule):
    def __init__(self, model, learning_rate, max_learning_rate, step_size_up, vocab_size):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.step_size_up = step_size_up
        self.max_learning_rate = max_learning_rate

        self.train_loss = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)  # Ignore padding index
        self.val_loss = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)  # Ignore padding index

        self.train_accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=PAD_TOKEN_ID)
        self.val_accuracy_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=PAD_TOKEN_ID)

        self.train_perplexity_metric = Perplexity(ignore_index=PAD_TOKEN_ID)
        self.val_perplexity_metric = Perplexity(ignore_index=PAD_TOKEN_ID)

    def configure_optimizers(self) -> OptimizerLRScheduler:
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

def get_quantizer() -> Int8DynActInt4WeightQATQuantizer:
    return Int8DynActInt4WeightQATQuantizer()

def get_training_model(params, checkpoint_path: Optional[str], device: str) -> Tuple[TrainingModel, Int8DynActInt4WeightQATQuantizer]:
    model = get_model(params, checkpoint_path, device)
    quantizer = get_quantizer()
    model = quantizer.prepare(model)

    return TrainingModel(model, learning_rate = params['learning_rate'], step_size_up=params['step_size_up'], max_learning_rate=params['max_learning_rate'], vocab_size=params['vocab_size']), quantizer
