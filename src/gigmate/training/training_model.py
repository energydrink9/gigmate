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
from torcheval.metrics import FrechetAudioDistance
from encodec.utils import save_audio

from gigmate.dataset.dataset import get_inputs_and_targets, restore_initial_sequence, DatasetBatch
from gigmate.domain.sampling import sample_from_logits
from gigmate.model.codec import decode
from gigmate.model.model import get_model
from gigmate.model.utils import compile_model
from gigmate.training.greedy_lr import GreedyLR
from gigmate.utils.constants import MAX_DECODER_SEQ_LEN, get_pad_token_id

PAD_TOKEN_ID = get_pad_token_id()
TEMP_DIR = tempfile.gettempdir()
NUMBER_OF_PREDICTED_SAMPLES_TO_KEEP = 10
FRECHET_AUDIO_DISTANCE_LENGTH = 128
AUDIO_SAMPLES_LENGTH = 500
# Weights for the codebooks in loss calculation
CODEBOOKS_LOSS_WEIGHTS: Dict[int, float] = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
}
CURRICULUM_LEARNING = False
# TODO: Enable compilation of training model
COMPILE_TRAINING_MODEL = False


# Define the weighted cross-entropy loss function
def weighted_cross_entropy_loss(logits, targets, weights):
    """
    Custom cross-entropy loss for curriculum learning with token-based weights.
    
    Args:
        logits (torch.Tensor): Model output logits of shape [batch_size, seq_length, vocab_size].
        targets (torch.Tensor): Ground truth labels of shape [batch_size, seq_length].
        weights (torch.Tensor): 1D tensor of shape [seq_length] with progressive weights for each token index.
        
    Returns:
        torch.Tensor: The computed weighted cross-entropy loss.
    """
    
    # Reshape weights to apply to each token in the sequence
    weights = weights.view(1, -1, 1)  # Shape [1, seq_length, 1]
    
    # Calculate cross-entropy loss for each token individually
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # Compute log probabilities
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # Gather log probabilities for target tokens
    
    # Compute weighted loss per token
    weighted_token_loss = -target_log_probs * weights  # Apply weights to each token
    loss = weighted_token_loss.mean()  # Average across batch and sequence
    
    return loss


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
    # flat_target, flat_pred = remove_special_tokens_from_target_and_logits(flat_target, flat_pred, get_special_tokens())

    # Decoding
    pred_audio, pred_audio_sr = decode(flat_pred, logit.device.type)
    target_audio, target_audio_sr = decode(flat_target, logit.device.type)

    return pred_audio, pred_audio_sr, target_audio, target_audio_sr


class TrainingModel(L.LightningModule):
    def __init__(self, model, learning_rate: float, max_learning_rate: float, vocab_size: int, codebooks: int, steps_per_epoch: int, logger: Optional[Logger]):
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
            warmup=8,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": greedy_lr_scheduler,
                "interval": "step",
                "reduce_on_plateau": True,
                "monitor": "train_loss_step",
                "strict": True,
            },
        }

    def log_metrics(self, dataset: Literal['train', 'val', 'test'], batch_size: Optional[int], interval: Literal['step', 'epoch'] = 'step', **kwargs):
        for (key, value) in kwargs.items():
            self.log_metric(dataset, batch_size, key, value, interval=interval)

    @torch.compiler.disable
    def log_metric(self, dataset: Optional[Literal['train', 'val', 'test']], batch_size: Optional[int], metric_name: str, value: Union[float, int], interval: Literal['step', 'epoch'] = 'step'):
        is_train_dataset = dataset == 'train'
        prog_bar = metric_name in ['loss', 'accuracy'] or (metric_name in ['perplexity', 'frechet_audio_distance'] and not is_train_dataset)
        metric_name_with_dataset = f"{dataset}_{metric_name}" if dataset is not None else metric_name
        self.log(metric_name_with_dataset, value, on_step=interval == 'step', on_epoch=True, prog_bar=prog_bar, batch_size=batch_size)

    def compute_loss(self, logits: Dict[int, Tensor], targets: Dict[int, Tensor], set: str, step: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        metrics: Dict[str, Tensor] = dict()
        total_loss = torch.tensor(0., device=self.device)

        if CURRICULUM_LEARNING:
            # logits have shape (B, V, S) == (B, 2048, 512)
            w = step / (torch.tensor(range(0, MAX_DECODER_SEQ_LEN), device=self.device.type) + 1)
            weights = torch.clamp(w, max=1)

            for k in range(self.codebooks):
                codebook_loss = weighted_cross_entropy_loss(logits[k], targets[k], weights)
                metrics[f'loss-{k}'] = codebook_loss

                weight = CODEBOOKS_LOSS_WEIGHTS[k]
                total_loss += codebook_loss * weight
            
            total_loss = total_loss / self.codebooks
            
            return total_loss, metrics
        
        else:

            for k in range(self.codebooks):
                
                codebook_loss = self.loss[set][k](logits[k], targets[k])
                metrics[f'loss-{k}'] = codebook_loss

                weight = CODEBOOKS_LOSS_WEIGHTS[k]
                total_loss += codebook_loss * weight
            
            total_loss = total_loss / self.codebooks
            
            return total_loss, metrics
        
    def compute_metrics(self, logits: Dict[int, Tensor], target: Dict[int, Tensor], logits_for_loss: Dict[int, Tensor], set: str) -> Dict[str, Tensor]:
        metrics = dict()
        accuracy_sum = torch.tensor(0., device=self.device)
        accuracy_real = torch.tensor(1., device=self.device)
        perplexity_sum = torch.tensor(0., device=self.device)

        for k in range(self.codebooks):
            accuracy = self.accuracy_metric[set][k].to(self.device)(logits_for_loss[k], target[k])
            accuracy_sum += accuracy
            accuracy_real *= accuracy
            
            perplexity = self.perplexity_metric[set][k].to(self.device)(logits[k], target[k])
            perplexity_sum += perplexity

            metrics[f'accuracy-{k}'] = accuracy
            metrics[f'perplexity-{k}'] = perplexity
        
        metrics['accuracy'] = accuracy_sum / self.codebooks
        metrics['accuracy_real'] = accuracy_real
        metrics['perplexity'] = perplexity_sum / self.codebooks

        return metrics
    
    def get_curriculum_learning_step(self) -> int:
        return int(self.global_step / self.steps_per_epoch * 4)

    def training_step(self, batch: DatasetBatch, batch_idx: int):
        curriculum_learning_step = self.get_curriculum_learning_step()
        self.log_metric(None, None, 'curriculum-learning', curriculum_learning_step, interval='step')
        
        full_track, stem, targets, sequence_lengths = get_inputs_and_targets(batch, self.device)
        logits, _, _ = self.model(stem, conditioning_input=full_track, sequence_lengths=sequence_lengths)

        codebook_logits, codebook_targets, codebook_logits_for_loss = get_codebook_logits_and_targets(self.codebooks, logits, targets)
        loss, loss_metrics = self.compute_loss(codebook_logits_for_loss, codebook_targets, 'train', curriculum_learning_step)
        metrics = self.compute_metrics(codebook_logits, codebook_targets, codebook_logits_for_loss, 'train')
        
        if batch_idx < 25:
            if sequence_lengths.stem[0] > FRECHET_AUDIO_DISTANCE_LENGTH:
                length = min(sequence_lengths.stem[0], FRECHET_AUDIO_DISTANCE_LENGTH)
                self.log_frechet_audio_distance_metric(batch.size, logits[:1, :, :length, :], targets[:1, :, :length], length, 'train')

        self.log_metrics('train', batch.size, interval='step', loss=loss, **loss_metrics)
        self.log_metrics('train', batch.size, interval='step', **metrics)

        return loss
    
    def validation_step(self, batch: DatasetBatch, batch_idx: int):
        curriculum_learning_step = self.get_curriculum_learning_step()
        full_track, stem, targets, sequence_lengths = get_inputs_and_targets(batch, self.device)
        logits, _, _ = self.model(stem, conditioning_input=full_track, sequence_lengths=sequence_lengths)

        codebook_logits, codebook_targets, codebook_logits_for_loss = get_codebook_logits_and_targets(self.codebooks, logits, targets)
        loss, loss_metrics = self.compute_loss(codebook_logits_for_loss, codebook_targets, 'val', curriculum_learning_step)
        metrics = self.compute_metrics(codebook_logits, codebook_targets, codebook_logits_for_loss, 'val')

        if batch_idx < 25:
            if sequence_lengths.stem[0] > FRECHET_AUDIO_DISTANCE_LENGTH:
                length = min(sequence_lengths.stem[0], FRECHET_AUDIO_DISTANCE_LENGTH)
                self.log_frechet_audio_distance_metric(batch.size, logits[:1, :, :length, :], targets[:1, :, :length], length, 'val')

            if batch_idx < 8:
                self.save_generated_audio(batch_idx, logits[:1, :, :AUDIO_SAMPLES_LENGTH, :], targets[:1, :, :AUDIO_SAMPLES_LENGTH], sequence_lengths.stem[0])

        self.log_metrics('val', batch.size, loss=loss, interval='step', **loss_metrics)
        self.log_metrics('val', batch.size, interval='step', **metrics)

        return loss

    def log_epoch_metrics(self, dataset_set: Literal['train', 'val', 'test']):
        accuracy_sum = 0
        accuracy_real = 1.
        perplexity_sum = 0

        for k in range(self.codebooks):
            accuracy_k = self.accuracy_metric[dataset_set][k].compute()
            perplexity_k = self.perplexity_metric[dataset_set][k].compute()
            
            accuracy_sum += accuracy_k
            accuracy_real *= accuracy_k
            perplexity_sum += perplexity_k
            
            self.log_metric(dataset_set, None, f'accuracy-{k}', accuracy_k, interval='epoch')
            self.log_metric(dataset_set, None, f'perplexity-{k}', perplexity_k, interval='epoch')

        accuracy = accuracy_sum / self.codebooks
        perplexity = perplexity_sum / self.codebooks

        self.log_metric(dataset_set, None, 'accuracy', accuracy, interval='epoch')
        self.log_metric(dataset_set, None, 'accuracy_real', accuracy_real, interval='epoch')
        self.log_metric(dataset_set, None, 'perplexity', perplexity, interval='epoch')

    def on_train_epoch_end(self):
        self.log_epoch_metrics('train')

    def on_validation_epoch_end(self):
        self.log_epoch_metrics('val')

    @torch.compiler.disable
    def log_frechet_audio_distance_metric(self, batch_size: int, logits: Tensor, targets: Tensor, sequence_length: int, dataset_set: Literal['train', 'val', 'test']):
        try:
            pred_audio, pred_audio_sr, target_audio, target_audio_sr = get_pred_and_target_audio(logits.detach(), targets.detach(), sequence_length)
            frechet_audio_distance_metric = self.frechet_audio_distance_metric['val']
            if self.device.type == 'cuda':  # fad metric is not supported on mps device
                frechet_audio_distance_metric = frechet_audio_distance_metric.to(self.device)
            
            # This is required to avoid errors related to mixed precision operations in fad calculation not being supported
            with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                frechet_audio_distance_metric.update(pred_audio.to(dtype=torch.float32), target_audio.to(dtype=torch.float32))
                frechet_audio_distance = frechet_audio_distance_metric.compute()
                
            self.log_metric(dataset_set, batch_size, 'frechet_audio_distance', frechet_audio_distance, interval='epoch')

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
            if self.task_logger is not None:
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


def get_training_model(params, checkpoint_path: Optional[str], device: str, task: Optional[Task], steps_per_epoch: int, compile=True) -> TrainingModel:
    model = get_model(params, checkpoint_path, device)

    training_model = TrainingModel(
        model,
        learning_rate=params['learning_rate'],
        max_learning_rate=params['max_learning_rate'],
        vocab_size=params['vocab_size'],
        codebooks=params['codebooks'],
        steps_per_epoch=steps_per_epoch,
        logger=task.get_logger() if task is not None else None
    )

    if compile is True and COMPILE_TRAINING_MODEL:
        training_model = cast(TrainingModel, compile_model(training_model, device_type=device))

    return training_model

