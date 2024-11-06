from statistics import mean
import warnings
from torch.optim.lr_scheduler import LRScheduler, _enable_get_lr_call
from collections import deque
from typing import Deque, List, Optional, cast
import torch
from torch import Tensor


class GreedyLR(LRScheduler):
    """
    GreedyLR scheduler that adaptively adjusts the learning rate based on loss improvements.
    The learning rate increases when loss improves and decreases when loss worsens.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        factor (float): Factor by which the learning rate will be increased/decreased. Default: 0.1
        patience (int): Number of steps with no improvement after which LR will be changed. Default: 10
        cooldown (int): Number of steps to wait before resuming normal operation after lr has been reduced. Default: 0
        warmup (int): Number of steps to wait before resuming normal operation after lr has been increased. Default: 0
        min_lr (float or list): A scalar or a list of lower bounds for learning rates. Default: 0
        max_lr (float or list): A scalar or a list of upper bounds for learning rates. Default: inf
        smooth (bool): Whether to use smoothed loss values. Default: True
        window (int): Window size for loss smoothing. Default: 50
        reset (int): Number of steps after which to reset the scheduler. Default: 0
        threshold (float): Threshold for measuring the new optimum. Default: 1e-4
        verbose (bool): If True, prints a message for each update. Default: False
    """
    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float,
                 total_steps: Optional[int] = None, factor: float = 0.1, 
                 patience: int = 10, cooldown: int = 0, warmup: int = 0,
                 min_lr: float = 0, max_lr: float = float('inf'), 
                 smooth: bool = True, window: int = 50, reset: int = 0,
                 threshold: float = 1e-4) -> None:

        for group in optimizer.param_groups:
            group['lr'] = initial_lr
            group.setdefault("initial_lr", initial_lr)

        if total_steps is None:
            raise Exception("total_steps is required")
        else:
            self.total_steps = total_steps

        self.factor = factor
        self.patience = patience
        self.cooldown = cooldown
        self.warmup = warmup
        self.smooth = smooth
        self.window = window
        self.reset_step = reset
        self.threshold = threshold
        
        # Convert learning rate bounds to lists
        self.min_lrs = [min_lr] * len(optimizer.param_groups) if not isinstance(min_lr, (list, tuple)) else min_lr
        self.max_lrs = [max_lr] * len(optimizer.param_groups) if not isinstance(max_lr, (list, tuple)) else max_lr
        
        # Initialize monitoring variables
        self.cooldown_counter = 0
        self.warmup_counter = 0
        self.num_good_steps = 0
        self.num_bad_steps = 0
        self.best_loss = float('inf')
        self.loss_window: Optional[Deque[float]] = deque(maxlen=window) if smooth else None
        self.current_loss: Optional[float] = None
        
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)
        
        return [group['lr'] for group in self.optimizer.param_groups]

    def get_updated_lr(self):

        lrs = []
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = group['lr']
            new_lr = old_lr
            
            if self.num_good_steps > self.patience:
                new_lr += old_lr * self.factor
                print(f'Increasing learning rate to: {new_lr}')
            elif self.num_bad_steps > self.patience:
                new_lr -= old_lr * self.factor
                print(f'Decreasing learning rate to: {new_lr}')
            
            # Clip learning rate
            new_lr = min(max(new_lr, self.min_lrs[i]), self.max_lrs[i])
            lrs.append(new_lr)

        return lrs

    def step(  # type: ignore[override] # noqa: C901
        self,
        metrics: Optional[Tensor] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Adjust the learning rate based on metrics.
        
        Args:
            metrics (Tensor, optional): Validation loss
            epoch (int, optional): The epoch number
        """
        # Increment epoch counter internally
        super().step(epoch)

        if self._step_count % self.total_steps == 0:
        
            # Early return if no metrics provided
            if metrics is None:
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
                return

            current_loss = cast(float, metrics.detach().cpu().item()) if metrics is not None else None
            self.current_loss = current_loss

            # Get smoothed loss if enabled
            if current_loss is not None:
                if self.smooth and self.loss_window is not None:
                    self.loss_window.append(current_loss)
                    current_loss = mean(list(self.loss_window))
                else:
                    current_loss = current_loss
            
            # Check if we should reset
            if self.reset_step > 0 and self._step_count > self.reset_step:
                self._reset()
                return

            # Update counters based on loss improvement
            if current_loss is not None:

                if current_loss < (self.best_loss - self.threshold):
                    self.best_loss = current_loss
                    self.num_good_steps += 1
                    self.num_bad_steps = 0
                else:
                    self.num_good_steps = 0
                    self.num_bad_steps += 1

            # Handle cooldown period
            if self.cooldown != 0:
                if self.cooldown_counter < self.cooldown:
                    self.cooldown_counter += 1
                else:
                    self.num_good_steps = 0
                    self.cooldown_counter = 0

            # Handle warmup period
            if self.warmup != 0:
                if self.warmup_counter < self.warmup:
                    self.warmup_counter += 1
                else:
                    self.num_bad_steps = 0
                    self.warmup_counter = 0

            # Update learning rates
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            
            with _enable_get_lr_call(self):
                new_lrs = self.get_updated_lr()
                
            for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
                param_group['lr'] = new_lr

    def _reset(self) -> None:
        """Reset the scheduler state."""
        self.best_loss = float('inf')
        self.num_good_steps = 0
        self.num_bad_steps = 0
        self.cooldown_counter = 0
        self.warmup_counter = 0
        
        if self.smooth and self.loss_window is not None:
            self.loss_window.clear()
            
        # Reset learning rates to initial values
        for param_group, initial_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = initial_lr
