import os
import tempfile
import traceback
from typing import Any, Dict, Optional, Tuple, Union, cast
from clearml import Logger, Task
from torchmetrics import Metric
import torchmetrics.classification
from torch import autocast, nn, Tensor
import lightning as L
from torchmetrics.text import Perplexity
import torch.optim
from typing import Literal
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchao.quantization.prototype.qat.api import Int8DynActInt4WeightQATQuantizer
from torcheval.metrics import FrechetAudioDistance
from encodec.utils import save_audio

from gigmate.dataset.dataset import get_inputs_and_targets, restore_initial_sequence, DatasetBatch
from gigmate.domain.sampling import sample_from_logits
from gigmate.model.codec import decode
from gigmate.model.model import ENABLE_QUANTIZATION, get_model
from gigmate.training.greedy_lr import GreedyLR
from gigmate.utils.constants import get_pad_token_id, get_special_tokens
from gigmate.utils.sequence_utils import remove_special_tokens_from_target_and_logits

PAD_TOKEN_ID = get_pad_token_id()
TEMP_DIR = tempfile.gettempdir()
NUMBER_OF_PREDICTED_SAMPLES_TO_KEEP = 10
FRECHET_AUDIO_DISTANCE_LENGTH = 128
AUDIO_SAMPLES_LENGTH = 500


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

    # Inverting interleaving and removing padding
    flat_pred = restore_initial_sequence(pred, sequence_length)
    flat_target = restore_initial_sequence(target, sequence_length)

    # Removing special tokens, as the decoder is not able to handle them
    flat_target, flat_pred = remove_special_tokens_from_target_and_logits(flat_target, flat_pred, get_special_tokens())

    # Decoding
    pred_audio, pred_audio_sr = decode(flat_pred, logit.device.type)
    target_audio, target_audio_sr = decode(flat_target, logit.device.type)

    return pred_audio, pred_audio_sr, target_audio, target_audio_sr


class TrainingModel(L.LightningModule):
    def __init__(self, model, learning_rate: float, max_learning_rate: float, vocab_size: int, codebooks: int, steps_per_epoch: int, logger: Logger):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.max_learning_rate = max_learning_rate
        self.codebooks = codebooks
        self.steps_per_epoch = steps_per_epoch
        
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

        # one_cycle_lr_scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=self.max_learning_rate,
        #     epochs=self.trainer.max_epochs,
        #     steps_per_epoch=self.steps_per_epoch,
        # )
        greedy_lr_scheduler = GreedyLR(
            optimizer,
            initial_lr=self.learning_rate,
            total_steps=self.steps_per_epoch // 16,
            max_lr=self.max_learning_rate,
            patience=2,
            window=6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": greedy_lr_scheduler,
                "interval": "step",
                "reduce_on_plateau": True,
                "monitor": "train_loss",
                "strict": True,
            },
        }

    def log_metrics(self, dataset: Literal['train', 'val', 'test'], batch_size: int, **kwargs):
        for (key, value) in kwargs.items():
            self.log_metric(dataset, batch_size, key, value)

    @torch.compiler.disable
    def log_metric(self, dataset: Literal['train', 'val', 'test'], batch_size: int, metric_name: str, value: Union[float, int]):
        is_train_dataset = dataset == 'train'
        on_step = metric_name in ['loss', 'accuracy']
        prog_bar = metric_name in ['loss', 'accuracy'] or (metric_name in ['perplexity', 'frechet_audio_distance'] and not is_train_dataset)
        self.log(f"{dataset}_{metric_name}", value, on_step=is_train_dataset and on_step, on_epoch=True, prog_bar=prog_bar, batch_size=batch_size)

    def compute_loss(self, logits: Dict[int, Tensor], targets: Dict[int, Tensor], set: str) -> Tuple[Tensor, Dict[str, Tensor]]:
        total_loss = torch.tensor(0., device=self.device)
        metrics: Dict[str, Tensor] = dict()

        for k in range(self.codebooks):
            codebook_loss = self.loss[set][k](logits[k], targets[k])
            metrics[f'loss-{k}'] = codebook_loss
            total_loss += codebook_loss
        
        total_loss = total_loss / (self.codebooks)
        
        return total_loss, metrics
    
    def compute_metrics(self, logits: Dict[int, Tensor], target: Dict[int, Tensor], logits_for_loss: Dict[int, Tensor], set: str) -> Dict[str, Tensor]:
        metrics = dict()
        accuracy_sum = torch.tensor(0., device=self.device)
        perplexity_sum = torch.tensor(0., device=self.device)

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
    
    def training_step(self, batch: DatasetBatch, batch_idx: int):
        full_track, stem, targets, sequence_lengths = get_inputs_and_targets(batch, self.device)
        logits, _, _ = self.model(stem, conditioning_input=full_track, sequence_lengths=sequence_lengths)

        codebook_logits, codebook_targets, codebook_logits_for_loss = get_codebook_logits_and_targets(self.codebooks, logits, targets)
        loss, loss_metrics = self.compute_loss(codebook_logits_for_loss, codebook_targets, 'train')
        metrics = self.compute_metrics(codebook_logits, codebook_targets, codebook_logits_for_loss, 'train')
        
        self.log_metrics('train', batch.size, loss=loss, **loss_metrics)
        self.log_metrics("train", batch.size, **metrics)

        return loss
    
    def validation_step(self, batch: DatasetBatch, batch_idx: int):
        full_track, stem, targets, sequence_lengths = get_inputs_and_targets(batch, self.device)
        logits, _, _ = self.model(stem, conditioning_input=full_track, sequence_lengths=sequence_lengths)

        codebook_logits, codebook_targets, codebook_logits_for_loss = get_codebook_logits_and_targets(self.codebooks, logits, targets)
        loss, loss_metrics = self.compute_loss(codebook_logits_for_loss, codebook_targets, 'val')
        metrics = self.compute_metrics(codebook_logits, codebook_targets, codebook_logits_for_loss, 'val')

        if batch_idx < 12:
            if sequence_lengths.stem[0] > FRECHET_AUDIO_DISTANCE_LENGTH:
                length = min(sequence_lengths.stem[0], FRECHET_AUDIO_DISTANCE_LENGTH)
                self.log_frechet_audio_distance_metric(batch.size, logits[:1, :, :length, :], targets[:1, :, :length], length)

            if batch_idx < 8:
                self.save_generated_audio(batch_idx, logits[:1, :, :AUDIO_SAMPLES_LENGTH, :], targets[:1, :, :AUDIO_SAMPLES_LENGTH], sequence_lengths.stem[0])

        self.log_metrics('val', batch.size, loss=loss, **loss_metrics)
        self.log_metrics('val', batch.size, **metrics)

        return loss

    @torch.compiler.disable
    def log_frechet_audio_distance_metric(self, batch_size: int, logits: Tensor, targets: Tensor, sequence_length: int):
        try:
            pred_audio, pred_audio_sr, target_audio, target_audio_sr = get_pred_and_target_audio(logits.detach(), targets.detach(), sequence_length)
            frechet_audio_distance_metric = self.frechet_audio_distance_metric['val']
            if self.device.type == 'cuda':  # fad metric is not supported on mps device
                frechet_audio_distance_metric = frechet_audio_distance_metric.to(self.device)
            
            # This is required to avoid errors related to mixed precision operations in fad calculation not being supported
            with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                frechet_audio_distance_metric.update(pred_audio.to(dtype=torch.float32), target_audio.to(dtype=torch.float32))
                frechet_audio_distance = frechet_audio_distance_metric.compute()
                
            self.log_metric('val', batch_size, 'frechet_audio_distance', frechet_audio_distance)

        except Exception as e:
            print('An error occurred while computing Frechet Audio Distance metric:')
            print(e)
            traceback.print_exc()

    @torch.compiler.disable
    def save_generated_audio(self, index: int, logits: Tensor, targets: Tensor, sequence_length: int):
        try:
            pred_audio, pred_audio_sr, target_audio, target_audio_sr = get_pred_and_target_audio(logits.detach(), targets.detach(), sequence_length)
            pred_audio = pred_audio.detach().to(device='cpu')
            target_audio = target_audio.detach().to(device='cpu')
            pred_audio_path = os.path.join(TEMP_DIR, 'pred_audio.wav')
            target_audio_path = os.path.join(TEMP_DIR, 'target_audio.wav')
            save_audio(path=pred_audio_path, wav=pred_audio.to(dtype=torch.float32), sample_rate=pred_audio_sr)
            save_audio(path=target_audio_path, wav=target_audio.to(dtype=torch.float32), sample_rate=target_audio_sr)
            self.task_logger.report_media(
                title='Predicted audio',
                series=f'{index}',
                iteration=self.current_epoch,
                local_path=pred_audio_path,
                file_extension='.wav',
                max_history=NUMBER_OF_PREDICTED_SAMPLES_TO_KEEP,
            )
            self.task_logger.report_media(
                title='Target audio',
                series=f'{index}',
                iteration=self.current_epoch,
                local_path=target_audio_path,
                file_extension='.wav',
                max_history=1,
            )
        except Exception as e:
            print('Error while generating predicted / target audio samples')
            print(e)
            traceback.print_exc()


def get_quantizer() -> Int8DynActInt4WeightQATQuantizer:
    return Int8DynActInt4WeightQATQuantizer()


def get_training_model(params, checkpoint_path: Optional[str], device: str, task: Task, steps_per_epoch: int) -> Tuple[TrainingModel, Optional[Int8DynActInt4WeightQATQuantizer]]:
    model = get_model(params, checkpoint_path, device)
    if device == 'cuda' and ENABLE_QUANTIZATION is True:
        quantizer = get_quantizer()
        model = quantizer.prepare(model)
    else:
        quantizer = None

    training_model = TrainingModel(
        model,
        learning_rate=params['learning_rate'],
        max_learning_rate=params['max_learning_rate'],
        vocab_size=params['vocab_size'],
        codebooks=params['codebooks'],
        steps_per_epoch=steps_per_epoch,
        logger=task.get_logger()
    )

    # TODO: fix torch compile full graph
    backend = 'aot_eager' if device == 'mps' or device == 'cuda' else 'inductor'
    training_model = cast(TrainingModel, torch.compile(training_model, fullgraph=False, backend=backend))
    
    return training_model, quantizer

