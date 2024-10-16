import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, cast
from clearml import Logger, Task
from torchmetrics import Metric
import torchmetrics.classification
from torch import nn, Tensor
import lightning as L
from torchmetrics.text import Perplexity
from torch.optim.lr_scheduler import CyclicLR
from typing import Literal
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchao.quantization.prototype.qat.api import Int8DynActInt4WeightQATQuantizer
from torcheval.metrics import FrechetAudioDistance
from encodec.utils import save_audio

from gigmate.dataset.dataset import get_inputs_and_targets, restore_initial_sequence
from gigmate.domain.sampling import sample_from_logits
from gigmate.model.codec import decode
from gigmate.model.model import ENABLE_QUANTIZATION, get_model
from gigmate.utils.constants import get_pad_token_id

PAD_TOKEN_ID = get_pad_token_id()
TEMP_DIR = tempfile.gettempdir()


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


def get_pred_and_target_audio(logit: Tensor, target: Tensor, sequence_length: int) -> Tuple[Tensor, int, Tensor, int]:

    # Prevent sampling special tokens as they would cause the decoder to fail
    pred = sample_from_logits(logit, 0., no_special_tokens=True)

    flat_pred = restore_initial_sequence(pred, sequence_length)
    flat_target = restore_initial_sequence(target, sequence_length)

    pred_audio, pred_audio_sr = decode(flat_pred, 'cpu')
    target_audio, target_audio_sr = decode(flat_target, 'cpu')

    return pred_audio, pred_audio_sr, target_audio, target_audio_sr


class TrainingModel(L.LightningModule):
    def __init__(self, model, learning_rate: float, max_learning_rate: float, step_size_up: int, vocab_size: int, codebooks: int, logger: Logger):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.step_size_up = step_size_up
        self.max_learning_rate = max_learning_rate
        self.codebooks = codebooks
        
        self.loss: Dict[str, Dict[int, nn.CrossEntropyLoss]] = dict({'train': dict(), 'val': dict()})
        self.accuracy_metric: Dict[str, Dict[int, Metric]] = dict({'train': dict(), 'val': dict()})
        self.perplexity_metric: Dict[str, Dict[int, Metric]] = dict({'train': dict(), 'val': dict()})
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
        self.task_logger = logger

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

    @torch.compiler.disable
    def log_metric(self, dataset: Literal['train', 'val', 'test'], metric_name: str, value):
        log_on_step = dataset == 'train'
        self.log(f"{dataset}_{metric_name}", value, on_step=log_on_step, on_epoch=True, prog_bar=metric_name in ['loss', 'accuracy', 'perplexity', 'frechet_audio_distance'])

    def compute_loss(self, logits: Dict[int, Tensor], targets: Dict[int, Tensor], set: str) -> Tuple[Tensor, Dict[str, Tensor]]:
        total_loss = torch.tensor(0.).to(self.device)
        metrics: Dict[str, Tensor] = dict()

        for k in range(self.codebooks):
            codebook_loss = self.loss[set][k](logits[k], targets[k])
            metrics[f'loss-{k}'] = codebook_loss
            total_loss += codebook_loss
        
        return total_loss, metrics
    
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
        full_track, stem, targets, sequence_lengths = get_inputs_and_targets(batch, self.device)
        logits, _ = self.model(stem, conditioning_input=full_track, sequence_lengths=sequence_lengths)

        codebook_logits, codebook_targets, codebook_logits_for_loss = get_codebook_logits_and_targets(self.codebooks, logits, targets)
        loss, loss_metrics = self.compute_loss(codebook_logits_for_loss, codebook_targets, 'train')
        metrics = self.compute_metrics(codebook_logits, codebook_targets, codebook_logits_for_loss, 'train')
        
        self.log_metrics('train', loss=loss, **loss_metrics)
        self.log_metrics("train", **metrics)

        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int):
        full_track, stem, targets, sequence_lengths = get_inputs_and_targets(batch, self.device)
        logits, _ = self.model(stem, conditioning_input=full_track, sequence_lengths=sequence_lengths)

        codebook_logits, codebook_targets, codebook_logits_for_loss = get_codebook_logits_and_targets(self.codebooks, logits, targets)
        loss, loss_metrics = self.compute_loss(codebook_logits_for_loss, codebook_targets, 'val')
        metrics = self.compute_metrics(codebook_logits, codebook_targets, codebook_logits_for_loss, 'val')

        if batch_idx < 10:
            self.log_frechet_audio_distance_metric(logits, targets, sequence_lengths)

        if batch_idx < 3:
            self.save_generated_audio(batch_idx, logits, targets, sequence_lengths)

        self.log_metrics('val', loss=loss, **loss_metrics)
        self.log_metrics('val', **metrics)

        return loss

    @torch.compiler.disable
    def log_frechet_audio_distance_metric(self, logits: Tensor, targets: Tensor, sequence_lengths: List[int]):
        pred_audio, pred_audio_sr, target_audio, target_audio_sr = get_pred_and_target_audio(logits[:1, :, :500, :].detach(), targets[:1, :, :500].detach(), sequence_lengths[0])
        frechet_audio_distance_metric = self.frechet_audio_distance_metric['val']
        if self.device.type == 'cuda':  # fad metric is not supported on mps device
            frechet_audio_distance_metric = frechet_audio_distance_metric.to(self.device)
        frechet_audio_distance_metric.update(pred_audio, target_audio)
        frechet_audio_distance = frechet_audio_distance_metric.compute()
        self.log_metric('val', 'frechet_audio_distance', frechet_audio_distance)

    @torch.compiler.disable
    def save_generated_audio(self, index: int, logits: Tensor, targets: Tensor, sequence_lengths: List[int]):
        try:
            pred_audio, pred_audio_sr, target_audio, target_audio_sr = get_pred_and_target_audio(logits[:1, :, :500, :].detach(), targets[:1, :, :500].detach(), sequence_lengths[0])
            pred_audio_path = os.path.join(TEMP_DIR, 'pred_audio.wav')
            target_audio_path = os.path.join(TEMP_DIR, 'target_audio.wav')
            save_audio(path=pred_audio_path, wav=pred_audio, sample_rate=pred_audio_sr)
            save_audio(path=target_audio_path, wav=target_audio, sample_rate=target_audio_sr)
            self.task_logger.report_media(
                title='Predicted audio',
                series=f'{index}',
                iteration=self.current_epoch,
                local_path=pred_audio_path,
                file_extension='.wav',
            )
            self.task_logger.report_media(
                title='Target audio',
                series=f'{index}',
                iteration=self.current_epoch,
                local_path=target_audio_path,
                file_extension='.wav',
            )
        except Exception as e:
            print('Error while generating predicted / target audio samples')
            print(e)


def get_quantizer() -> Int8DynActInt4WeightQATQuantizer:
    return Int8DynActInt4WeightQATQuantizer()


def get_training_model(params, checkpoint_path: Optional[str], device: str, task: Task) -> Tuple[TrainingModel, Optional[Int8DynActInt4WeightQATQuantizer]]:
    model = get_model(params, checkpoint_path, device)
    if device == 'cuda' and ENABLE_QUANTIZATION is True:
        quantizer = get_quantizer()
        model = quantizer.prepare(model)
    else:
        quantizer = None

    training_model = TrainingModel(
        model,
        learning_rate=params['learning_rate'],
        step_size_up=params['step_size_up'],
        max_learning_rate=params['max_learning_rate'],
        vocab_size=params['vocab_size'],
        codebooks=params['codebooks'],
        logger=task.get_logger()
    )

    # TODO: fix torch compile full graph
    backend = 'aot_eager' if device == 'mps' or device == 'cuda' else 'inductor'
    training_model = cast(TrainingModel, torch.compile(training_model, fullgraph=False, backend=backend))
    
    return training_model, quantizer

