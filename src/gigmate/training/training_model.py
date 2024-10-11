from typing import Any, Dict, Optional, Tuple
from torchmetrics import Metric
import torchmetrics.classification
from torch import nn, Tensor
import lightning as L
from torchmetrics.text import Perplexity
from torch.optim.lr_scheduler import CyclicLR
from typing import Literal
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from gigmate.dataset.dataset import get_inputs_and_targets, restore_initial_sequence
from gigmate.domain.sampling import remove_forbidden_tokens, sample_from_logits
from gigmate.model.codec import decode
from gigmate.model.model import get_model
from torchao.quantization.prototype.qat.api import Int8DynActInt4WeightQATQuantizer
from torcheval.metrics import FrechetAudioDistance

from gigmate.utils.constants import get_pad_token_id

PAD_TOKEN_ID = get_pad_token_id()


def reshape_logits_for_loss_calculation(logits: Tensor) -> Tensor:
    # This formats are equivalent:
    # - `transposed_logits = logits.permute(0, 2, 1)`
    # - `logits.transpose(1, 2)`
    return logits.permute(0, 2, 1)


def get_codebook_logits_and_targets(codebooks: int, logits: Tensor, targets: Tensor) -> Tuple[Dict[int, Tensor], Dict[int, Tensor], Dict[int, Tensor]]:
    codebook_logits = dict()
    codebook_targets = dict()
    logits_for_loss = dict()

    for k in range(codebooks):
        codebook_logits[k] = logits[:, k, :, :].squeeze(1)
        codebook_targets[k] = targets[:, k, :].squeeze(1)
        logits_for_loss[k] = reshape_logits_for_loss_calculation(codebook_logits[k])

    return codebook_logits, codebook_targets, logits_for_loss


def get_pred_and_target_audio(logit: Tensor, target: Tensor, sequence_length: int) -> Tuple[Tensor, Tensor]:

    # Prevent sampling special tokens as they would cause the decoder to fail
    pred = sample_from_logits(logit, 0., no_special_tokens=True)

    flat_pred = restore_initial_sequence(pred, sequence_length)
    flat_target = restore_initial_sequence(target, sequence_length)

    pred_audio = decode(flat_pred, 'cpu')
    target_audio = decode(flat_target, 'cpu')

    return pred_audio, target_audio


class TrainingModel(L.LightningModule):
    def __init__(self, model, learning_rate: float, max_learning_rate: float, step_size_up: int, vocab_size: int, codebooks: int):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.step_size_up = step_size_up
        self.max_learning_rate = max_learning_rate
        self.codebooks = codebooks
        
        self.loss: Dict[str, Dict[int, nn.CrossEntropyLoss]] = dict({ 'train': dict(), 'val': dict() })
        self.accuracy_metric: Dict[str, Dict[int, Metric]] = dict({ 'train': dict(), 'val': dict() })
        self.perplexity_metric: Dict[str, Dict[int, Metric]] = dict({ 'train': dict(), 'val': dict() })
        self.frechet_audio_distance_metric: Dict[str, Any] = dict()
        
        for k in range(codebooks):
            self.loss['train'][k] = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)  # Ignore padding index
            self.loss['val'][k] = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)  # Ignore padding index

        for k in range(codebooks):
            self.accuracy_metric['train'][k] = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=PAD_TOKEN_ID)
            self.accuracy_metric['val'][k] = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=PAD_TOKEN_ID)

        for k in range(codebooks):
            self.perplexity_metric['train'][k] = Perplexity(ignore_index=PAD_TOKEN_ID)
            self.perplexity_metric['val'][k] = Perplexity(ignore_index=PAD_TOKEN_ID)

        self.frechet_audio_distance_metric['train'] = FrechetAudioDistance.with_vggish()
        self.frechet_audio_distance_metric['val'] = FrechetAudioDistance.with_vggish()

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

    def log_metrics(self, dataset: Literal['train', 'val', 'test'], **kwargs):
        for (key, value) in kwargs.items():
            self.log_metric(dataset, key, value)

    def log_metric(self, dataset: Literal['train', 'val', 'test'], metric_name: str, value):
        log_on_step = dataset == 'train'
        self.log(f"{dataset}_{metric_name}", value, on_step=log_on_step, on_epoch=True, prog_bar=metric_name in ['loss', 'accuracy', 'perplexity', 'frechet_audio_distance'])

    def compute_loss(self, logits: Dict[int, Tensor], targets: Dict[int, Tensor], set: str) -> Tensor:
        total_loss = torch.tensor(0.).to(self.device)

        for k in range(self.codebooks):
            codebook_loss = self.loss[set][k](logits[k], targets[k])
            total_loss += codebook_loss
        
        return total_loss
    
    def compute_metrics(self, logits: Dict[int, Tensor], target: Dict[int, Tensor], logits_for_loss: Dict[int, Tensor], set: str) -> Dict[str, Tensor]:
        metrics = dict()
        accuracy_sum = torch.tensor(0.).to(self.device)
        perplexity_sum = torch.tensor(0.).to(self.device)

        for k in range(self.codebooks):
            accuracy = self.accuracy_metric[set][k].to(self.device)(logits_for_loss[k], target[k])
            accuracy_sum += accuracy

            perplexity = self.perplexity_metric[set][k].to(self.device)(logits[k], target[k])
            perplexity_sum += perplexity

            metrics[f'accuracy-{k}'] = accuracy
            metrics[f'perplexity-{k}'] = perplexity
            
        metrics['accuracy'] = accuracy_sum / self.codebooks
        metrics['perplexity'] = perplexity_sum / self.codebooks

        return metrics
    
    def training_step(self, batch: Tensor, batch_idx: int):
        inputs, targets, sequence_lengths = get_inputs_and_targets(batch, self.device)
        logits, _ = self.model(inputs, sequence_lengths)

        codebook_logits, codebook_targets, codebook_logits_for_loss = get_codebook_logits_and_targets(self.codebooks, logits, targets)
        loss = self.compute_loss(codebook_logits_for_loss, codebook_targets, 'train')

        metrics = self.compute_metrics(codebook_logits, codebook_targets, codebook_logits_for_loss, 'train')
        self.log_metrics("train", loss=loss, **metrics)

        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int):
        inputs, targets, sequence_lengths = get_inputs_and_targets(batch, self.device)
        logits, _ = self.model(inputs, sequence_lengths)

        codebook_logits, codebook_targets, codebook_logits_for_loss = get_codebook_logits_and_targets(self.codebooks, logits, targets)
        loss = self.compute_loss(codebook_logits_for_loss, codebook_targets, 'val')

        metrics = self.compute_metrics(codebook_logits, codebook_targets, codebook_logits_for_loss, 'val')

        pred_audio, target_audio = get_pred_and_target_audio(logits[:1, :, :500, :].detach(), targets[:1, :, :500].detach(), sequence_lengths[0])

        if batch_idx < 32:
            frechet_audio_distance_metric = self.frechet_audio_distance_metric['train']
            if self.device == 'cuda': # fad metric is not supported on mps device
                frechet_audio_distance_metric = frechet_audio_distance_metric.to(self.device)
            frechet_audio_distance_metric.update(pred_audio, target_audio)
            frechet_audio_distance = frechet_audio_distance_metric.compute()
            metrics['frechet_audio_distance'] = frechet_audio_distance

        self.log_metrics("val", loss=loss, **metrics)

        return loss


def get_quantizer() -> Int8DynActInt4WeightQATQuantizer:
    return Int8DynActInt4WeightQATQuantizer()


def get_training_model(params, checkpoint_path: Optional[str], device: str) -> Tuple[TrainingModel, Optional[Int8DynActInt4WeightQATQuantizer]]:
    model = get_model(params, checkpoint_path, device)
    if device == 'cuda':
        quantizer = get_quantizer()
        model = quantizer.prepare(model)
    else:
        quantizer = None

    return TrainingModel(model, learning_rate = params['learning_rate'], step_size_up=params['step_size_up'], max_learning_rate=params['max_learning_rate'], vocab_size=params['vocab_size'], codebooks=params['codebooks']), quantizer

