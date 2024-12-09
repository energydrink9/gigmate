import math
import os
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from clearml import Logger, Task
from torchmetrics import Metric
import torchmetrics.classification
from torch import autocast, nn, Tensor
import lightning as L
from torchmetrics.text import Perplexity
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR
from typing import Literal
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torcheval.metrics import FrechetAudioDistance
from encodec.utils import save_audio

from gigmate.dataset.dataset import get_inputs_and_targets, get_shift, restore_initial_sequence, DatasetBatch
from gigmate.domain.prediction import complete_sequence
from gigmate.domain.sampling import sample_from_logits
from gigmate.model.codec import decode
from gigmate.model.model import get_model
from gigmate.model.utils import compile_model
from gigmate.training.greedy_lr import GreedyLR
from gigmate.utils.constants import MAX_DECODER_SEQ_LEN, get_pad_token_id
from gigmate.utils.env import get_environment
from gigmate.utils.sequence_utils import cut_sequence

ENVIRONMENT = get_environment()
PAD_TOKEN_ID = get_pad_token_id()
TEMP_DIR = tempfile.gettempdir()
NUMBER_OF_PREDICTED_SAMPLES_TO_KEEP = 10
FRECHET_AUDIO_DISTANCE_LENGTH = 512
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
FRAME_RATE = 50
NUMBER_OF_SAMPLES_FOR_FAD = 5 if ENVIRONMENT != 'dev' else 0
TEMPERATURE_HIGH = 0.8
TEMPERATURE_LOW = 0.4
LOG_FRECHET_AUDIO_DISTANCE_IN_TRAINING = False


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
        
    loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=PAD_TOKEN_ID, reduction='none')
    weighted_token_loss = loss * weights  # Apply weights
    
    return weighted_token_loss.mean()  # Average across batch and sequence
    

class LossByCodebookAndTokenPosition():
    values: List[Tensor]

    def __init__(self):
        self.reset()

    def update(self, value: Tensor) -> None:
        self.values.append(value)

    def compute(self, codebook: Optional[int]) -> List[float]:
        data = torch.vstack(self.values)

        if codebook is None:
            data = data.mean(dim=0)
        else:
            data = data[codebook, :]

        return data.detach().cpu().numpy().tolist()
    
    def reset(self):
        self.values = []


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


@torch.compiler.disable
def get_pred_audio(logit: Tensor, sequence_length: int, sample: bool = False) -> Tuple[Tensor, int]:

    pred = sample_from_logits(logit, 0., no_special_tokens=False)

    # Inverting interleaving and removing padding
    flat_pred = restore_initial_sequence(pred, sequence_length)

    # Decoding
    pred_audio, pred_audio_sr = decode(flat_pred, logit.device.type, to_cpu=True)

    return pred_audio.detach(), pred_audio_sr


def get_full_track_prediction_input(sequence: Tensor, sequence_length: int) -> Tensor:
    sequence = sequence[:1, :, :]
    sequence = cut_sequence(sequence, sequence_length, cut_left=True)
    return sequence
    # return revert_interleaving(sequence)


@torch.compiler.disable
def get_target_audio(target: Tensor, sequence_length: int) -> Tuple[Tensor, int]:

    # Inverting interleaving and removing padding
    flat_target = restore_initial_sequence(target, sequence_length)

    # Decoding
    target_audio, target_audio_sr = decode(flat_target, target.device.type, to_cpu=True)

    return target_audio.detach(), target_audio_sr


class TrainingModel(L.LightningModule):
    def __init__(self, model, learning_rate: float, max_learning_rate: float, min_learning_rate: float, vocab_size: int, codebooks: int, steps_per_epoch: int, logger: Optional[Logger]):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.codebooks = codebooks
        self.steps_per_epoch = steps_per_epoch
        
        self.loss: Dict[str, Dict[int, nn.CrossEntropyLoss]] = dict({'train': dict(), 'val': dict()})
        self.loss_by_codebook_and_token_position: Dict[str, LossByCodebookAndTokenPosition] = dict({'train': LossByCodebookAndTokenPosition(), 'val': LossByCodebookAndTokenPosition()})
        self.accuracy_metric: Dict[str, Dict[int, Metric]] = dict({'train': dict(), 'val': dict()})
        self.perplexity_metric: Dict[str, Dict[int, Metric]] = dict({'train': dict(), 'val': dict()})
        self.frechet_audio_distance_metric_tf: Dict[str, Any] = dict()
        self.frechet_audio_distance_metric: Dict[str, Any] = dict()
        self.frechet_audio_distance_metric_temp_high: Dict[str, Any] = dict()
        self.frechet_audio_distance_metric_temp_low: Dict[str, Any] = dict()

        for k in range(codebooks):
            self.loss['train'][k] = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='none')  # Ignore padding index
            self.loss['val'][k] = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='none')  # Ignore padding index

        for k in range(codebooks):
            self.accuracy_metric['train'][k] = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=PAD_TOKEN_ID)
            self.accuracy_metric['val'][k] = torchmetrics.classification.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=PAD_TOKEN_ID)

        for k in range(codebooks):
            self.perplexity_metric['train'][k] = Perplexity(ignore_index=PAD_TOKEN_ID)
            self.perplexity_metric['val'][k] = Perplexity(ignore_index=PAD_TOKEN_ID)

        self.frechet_audio_distance_metric_tf['train'] = FrechetAudioDistance.with_vggish()
        self.frechet_audio_distance_metric_tf['val'] = FrechetAudioDistance.with_vggish()
        # self.frechet_audio_distance_metric['train'] = FrechetAudioDistance.with_vggish()
        # self.frechet_audio_distance_metric['val'] = FrechetAudioDistance.with_vggish()
        self.frechet_audio_distance_metric_temp_high['train'] = FrechetAudioDistance.with_vggish()
        self.frechet_audio_distance_metric_temp_high['val'] = FrechetAudioDistance.with_vggish()
        self.frechet_audio_distance_metric_temp_low['train'] = FrechetAudioDistance.with_vggish()
        self.frechet_audio_distance_metric_temp_low['val'] = FrechetAudioDistance.with_vggish()

        self.task_logger = logger

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        schedulers = {
            'greedy': GreedyLR(
                optimizer,
                initial_lr=self.learning_rate,
                total_steps=self.steps_per_epoch // 16,
                max_lr=self.max_learning_rate,
                min_lr=self.min_learning_rate,
                patience=5,
                window=25,
                warmup=15,
            ),
            'one_cycle': OneCycleLR(
                optimizer,
                max_lr=self.max_learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.steps_per_epoch,
                pct_start=0.04,
            )
        }

        scheduler = schedulers['one_cycle']

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "reduce_on_plateau": scheduler == schedulers['greedy'],
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

    def compute_loss(self, logits: Dict[int, Tensor], targets: Dict[int, Tensor], set: str) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:

        metrics: Dict[str, Tensor] = dict()
        total_loss = torch.tensor(0., device=self.device)

        if CURRICULUM_LEARNING:
            # logits have shape (B, V, S) == (B, 2048, 512)
            # max(e^-(x/100), 0.1)
            w = math.e ** -(torch.tensor(range(0, MAX_DECODER_SEQ_LEN), device=self.device.type) / 100)
            weights = torch.clamp(w, min=0.10, max=1)

            for k in range(self.codebooks):
                codebook_loss = weighted_cross_entropy_loss(logits[k], targets[k], weights)
                metrics[f'loss-{k}'] = codebook_loss

                weight = CODEBOOKS_LOSS_WEIGHTS[k]
                total_loss += codebook_loss * weight
            
            total_loss = total_loss / self.codebooks * 512 / (90 + 0.10 * (512 - 230))
            
            return total_loss, metrics, torch.tensor([])
        
        else:

            codebooks_loss_by_position = []

            for k in range(self.codebooks):
                
                codebook_loss_unreduced: Tensor = self.loss[set][k](logits[k], targets[k])
                codebook_loss = codebook_loss_unreduced.mean()
                codebooks_loss_by_position.append(codebook_loss_unreduced.mean(dim=0))
                metrics[f'loss-{k}'] = codebook_loss

                weight = CODEBOOKS_LOSS_WEIGHTS[k]
                total_loss += codebook_loss * weight
            
            total_loss = total_loss / self.codebooks
            loss_by_codebooks_position = torch.vstack(codebooks_loss_by_position)
            
            return total_loss, metrics, loss_by_codebooks_position
        
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
        # curriculum_learning_step = self.get_curriculum_learning_step()
        # self.log_metric(None, None, 'curriculum-learning', curriculum_learning_step, interval='step')
        
        full_track, stem, targets, sequence_lengths = get_inputs_and_targets(batch, self.device)
        logits, _, _ = self.model(stem, conditioning_input=full_track, sequence_lengths=sequence_lengths, shift=get_shift(MAX_DECODER_SEQ_LEN))

        codebook_logits, codebook_targets, codebook_logits_for_loss = get_codebook_logits_and_targets(self.codebooks, logits, targets)
        loss, loss_metrics, loss_by_codebooks_position = self.compute_loss(codebook_logits_for_loss, codebook_targets, 'train')
        self.loss_by_codebook_and_token_position['train'].update(loss_by_codebooks_position)
        metrics = self.compute_metrics(codebook_logits, codebook_targets, codebook_logits_for_loss, 'train')
        
        if LOG_FRECHET_AUDIO_DISTANCE_IN_TRAINING and batch_idx < NUMBER_OF_SAMPLES_FOR_FAD:
            if sequence_lengths.stem[0] >= FRECHET_AUDIO_DISTANCE_LENGTH:
                length = min(sequence_lengths.stem[0], FRECHET_AUDIO_DISTANCE_LENGTH)
                logits_fad = logits[:1, :, :length, :]
                targets_fad = targets[:1, :, :length]
                pred_audio_tf, pred_audio_tf_sr = get_pred_audio(logits_fad.detach().to(self.device), length)
                full_track_prediction_input = get_full_track_prediction_input(full_track[:1, :, :], sequence_lengths.full_track[0])

                pred_audio_temp_high_encoded = self.predict_continuation(full_track_prediction_input, length, TEMPERATURE_HIGH)
                pred_audio_temp_high, pred_audio_temp_high_sr = decode(pred_audio_temp_high_encoded, full_track.device.type, to_cpu=True)
                pred_audio_temp_low_encoded = self.predict_continuation(full_track_prediction_input, length, TEMPERATURE_LOW)
                pred_audio_temp_low, pred_audio_temp_low_sr = decode(pred_audio_temp_low_encoded, full_track.device.type, to_cpu=True)
                target_audio, target_audio_sr = get_target_audio(targets_fad.detach(), length)
                self.log_frechet_audio_distance_metric(batch.size, pred_audio_tf, target_audio, 'train', self.frechet_audio_distance_metric_tf['train'], 'tf')
                self.log_frechet_audio_distance_metric(batch.size, pred_audio_temp_high, target_audio, 'train', self.frechet_audio_distance_metric_temp_high['train'], 'temp_high')
                self.log_frechet_audio_distance_metric(batch.size, pred_audio_temp_low, target_audio, 'train', self.frechet_audio_distance_metric_temp_low['train'], 'temp_low')

                if batch_idx < 8:
                    self.save_generated_audio(
                        batch_idx,
                        pred_audio_tf,
                        pred_audio_temp_high,
                        pred_audio_temp_low,
                        target_audio,
                        pred_audio_tf_sr,
                        pred_audio_temp_high_sr,
                        pred_audio_temp_low_sr,
                        target_audio_sr,
                        'train'
                    )

        self.log_metrics('train', batch.size, interval='step', loss=loss, **loss_metrics)
        self.log_metrics('train', batch.size, interval='step', **metrics)

        return loss
    
    def predict_continuation(self, full_track: Tensor, length: int, temperature: float):
        return complete_sequence(self.model, full_track.device.type, full_track, FRAME_RATE, PAD_TOKEN_ID, length / FRAME_RATE, temperature, use_cache=False)

    def validation_step(self, batch: DatasetBatch, batch_idx: int):

        full_track, stem, targets, sequence_lengths = get_inputs_and_targets(batch, self.device)
        logits, _, _ = self.model(stem, conditioning_input=full_track, sequence_lengths=sequence_lengths, shift=get_shift(MAX_DECODER_SEQ_LEN))

        codebook_logits, codebook_targets, codebook_logits_for_loss = get_codebook_logits_and_targets(self.codebooks, logits, targets)
        loss, loss_metrics, loss_by_codebooks_position = self.compute_loss(codebook_logits_for_loss, codebook_targets, 'val')
        self.loss_by_codebook_and_token_position['val'].update(loss_by_codebooks_position)
        metrics = self.compute_metrics(codebook_logits, codebook_targets, codebook_logits_for_loss, 'val')

        if batch_idx < NUMBER_OF_SAMPLES_FOR_FAD:
            if sequence_lengths.stem[0] >= FRECHET_AUDIO_DISTANCE_LENGTH:
                length = min(sequence_lengths.stem[0], FRECHET_AUDIO_DISTANCE_LENGTH)
                logits_fad = logits[:1, :, :length, :]
                targets_fad = targets[:1, :, :length]
                pred_audio_tf, pred_audio_tf_sr = get_pred_audio(logits_fad.detach(), length)
                full_track_prediction_input = get_full_track_prediction_input(full_track[:1, :, :], sequence_lengths.full_track[0])
                pred_audio_temp_high_encoded = self.predict_continuation(full_track_prediction_input, length, TEMPERATURE_HIGH)
                pred_audio_temp_high, pred_audio_temp_high_sr = decode(pred_audio_temp_high_encoded, full_track.device.type, to_cpu=True)
                pred_audio_temp_low_encoded = self.predict_continuation(full_track_prediction_input, length, TEMPERATURE_LOW)
                pred_audio_temp_low, pred_audio_temp_low_sr = decode(pred_audio_temp_low_encoded, full_track.device.type, to_cpu=True)
                target_audio, target_audio_sr = get_target_audio(targets_fad.detach(), length)
                self.log_frechet_audio_distance_metric(batch.size, pred_audio_tf, target_audio, 'val', self.frechet_audio_distance_metric_tf['val'], 'tf')
                self.log_frechet_audio_distance_metric(batch.size, pred_audio_temp_high, target_audio, 'val', self.frechet_audio_distance_metric_temp_high['val'], 'temp_high')
                self.log_frechet_audio_distance_metric(batch.size, pred_audio_temp_low, target_audio, 'val', self.frechet_audio_distance_metric_temp_low['val'], 'temp_low')

                if batch_idx < 8:
                    self.save_generated_audio(batch_idx, pred_audio_tf, pred_audio_temp_high, pred_audio_temp_low, target_audio, pred_audio_tf_sr, pred_audio_temp_high_sr, pred_audio_temp_low_sr, target_audio_sr, 'val')

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

    def plot_loss_by_token_position(self, dataset_set: Literal['train', 'val'], loss_by_codebook_and_token_position: LossByCodebookAndTokenPosition, codebook: Optional[int]):
        if self.task_logger is not None:
            if codebook is None:
                self.task_logger.report_histogram(
                    title=f"Loss by token position [{dataset_set}]",
                    series="All codebooks",
                    iteration=self.current_epoch,
                    values=loss_by_codebook_and_token_position.compute(None),
                    xaxis="Token position",
                    yaxis="Loss",
                )
            else:
                self.task_logger.report_histogram(
                    title=f"Loss by token position [{dataset_set}]",
                    series=f"Codebook {codebook}",
                    iteration=self.current_epoch,
                    values=loss_by_codebook_and_token_position.compute(codebook),
                    xaxis="Token position",
                    yaxis="Loss",
                )

    def on_train_epoch_end(self):
        self.log_epoch_metrics('train')

        self.plot_loss_by_token_position('train', self.loss_by_codebook_and_token_position['train'], None)
        
        for codebook in range(self.codebooks):
            self.plot_loss_by_token_position('train', self.loss_by_codebook_and_token_position['train'], codebook)

        self.loss_by_codebook_and_token_position['train'].reset()

    def on_validation_epoch_end(self):
        self.log_epoch_metrics('val')

        self.plot_loss_by_token_position('val', self.loss_by_codebook_and_token_position['val'], None)

        for codebook in range(self.codebooks):
            self.plot_loss_by_token_position('val', self.loss_by_codebook_and_token_position['val'], codebook)

        self.loss_by_codebook_and_token_position['val'].reset()

    def log_frechet_audio_distance_metric(self, batch_size: int, pred_audio: Tensor, target_audio: Tensor, dataset_set: Literal['train', 'val', 'test'], metric: FrechetAudioDistance, name: Optional[str] = None):
        try:
            if self.device.type == 'cuda':  # fad metric is not supported on mps device
                metric = metric.to(self.device)
            
            # This is required to avoid errors related to mixed precision operations in fad calculation not being supported
            with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                metric.update(pred_audio.to(dtype=torch.float32), target_audio.to(dtype=torch.float32))
                frechet_audio_distance = metric.compute()
            
            self.log_metric(dataset_set, batch_size, f'frechet_audio_distance{"" if name is None else f"_{name}"}', frechet_audio_distance.item(), interval='epoch')

        except Exception as e:
            print('An error occurred while computing Frechet Audio Distance metric:')
            print(e)
            traceback.print_exc()

    @torch.compiler.disable
    def save_generated_audio(
        self,
        index: int,
        pred_audio_tf: Tensor,
        pred_audio_temp_high: Tensor,
        pred_audio_temp_low: Tensor,
        target_audio: Tensor,
        pred_audio_tf_sr: int,
        pred_audio_temp_high_sr: int,
        pred_audio_temp_low_sr: int,
        target_audio_sr: int,
        dataset_set: Literal['train', 'val', 'test']
    ):
        
        try:
            pred_audio_tf_path = os.path.join(TEMP_DIR, 'pred_audio_tf.wav')
            pred_audio_temp_high_path = os.path.join(TEMP_DIR, 'pred_audio_temp_high.wav')
            pred_audio_temp_low_path = os.path.join(TEMP_DIR, 'pred_audio_temp_low.wav')
            target_audio_path = os.path.join(TEMP_DIR, 'target_audio.wav')
            save_audio(path=pred_audio_tf_path, wav=pred_audio_tf.to(dtype=torch.float32), sample_rate=pred_audio_tf_sr)
            save_audio(path=pred_audio_temp_high_path, wav=pred_audio_temp_high.to(dtype=torch.float32), sample_rate=pred_audio_temp_high_sr)
            save_audio(path=pred_audio_temp_low_path, wav=pred_audio_temp_low.to(dtype=torch.float32), sample_rate=pred_audio_temp_low_sr)
            save_audio(path=target_audio_path, wav=target_audio.to(dtype=torch.float32), sample_rate=target_audio_sr)
            if self.task_logger is not None:
                self.task_logger.report_media(
                    title=f'Predicted audio TF ({dataset_set})',
                    series=f'{index}',
                    iteration=self.current_epoch,
                    local_path=pred_audio_tf_path,
                    file_extension='.wav',
                    max_history=NUMBER_OF_PREDICTED_SAMPLES_TO_KEEP,
                )
                self.task_logger.report_media(
                    title=f'Predicted audio Temp high ({dataset_set})',
                    series=f'{index}',
                    iteration=self.current_epoch,
                    local_path=pred_audio_temp_high_path,
                    file_extension='.wav',
                    max_history=NUMBER_OF_PREDICTED_SAMPLES_TO_KEEP,
                )
                self.task_logger.report_media(
                    title=f'Predicted audio Temp low ({dataset_set})',
                    series=f'{index}',
                    iteration=self.current_epoch,
                    local_path=pred_audio_temp_low_path,
                    file_extension='.wav',
                    max_history=NUMBER_OF_PREDICTED_SAMPLES_TO_KEEP,
                )
                self.task_logger.report_media(
                    title=f'Target audio ({dataset_set})',
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


def get_training_model(params, checkpoint_path: Optional[str], device: str, task: Optional[Task], steps_per_epoch: int, compile: bool = True) -> TrainingModel:
    model = get_model(params, checkpoint_path, device, compile=compile)

    training_model = TrainingModel(
        model,
        learning_rate=params['learning_rate'],
        max_learning_rate=params['max_learning_rate'],
        min_learning_rate=params['min_learning_rate'],
        vocab_size=params['vocab_size'],
        codebooks=params['codebooks'],
        steps_per_epoch=steps_per_epoch,
        logger=task.get_logger() if task is not None else None
    )

    if compile is True and COMPILE_TRAINING_MODEL:
        training_model = cast(TrainingModel, compile_model(training_model, device_type=device))

    return training_model

