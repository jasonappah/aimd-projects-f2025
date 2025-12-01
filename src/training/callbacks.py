"""Training callbacks: early stopping, checkpointing, learning rate scheduling."""

import torch
import os
from typing import Optional, Dict, Any
from pathlib import Path


class EarlyStopping:
    """Early stopping callback based on validation metric."""
    
    def __init__(self,
                 patience: int = 10,
                 mode: str = 'max',
                 min_delta: float = 0.0,
                 metric_name: str = 'pr_auc'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            min_delta: Minimum change to qualify as improvement
            metric_name: Name of metric to monitor
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric_value: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = metric_value
        elif self._is_better(metric_value, self.best_score):
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current value is better than best."""
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta


class ModelCheckpoint:
    """Model checkpointing callback."""
    
    def __init__(self,
                 checkpoint_dir: str,
                 save_best: bool = True,
                 save_last: bool = True,
                 metric_name: str = 'pr_auc',
                 mode: str = 'max'):
        """
        Initialize model checkpointing.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model
            save_last: Whether to save last model
            metric_name: Metric to use for best model selection
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_last = save_last
        self.metric_name = metric_name
        self.mode = mode
        self.best_score = None
        self.best_path = None
    
    def save(self,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             epoch: int,
             metric_value: float,
             metrics: Dict[str, float],
             is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metric_value: Current metric value
            metrics: Dictionary of all metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric_value': metric_value,
            'metrics': metrics
        }
        
        # Save last checkpoint
        if self.save_last:
            last_path = self.checkpoint_dir / 'checkpoint_last.pth'
            torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if self.save_best and is_best:
            if self.best_path and self.best_path.exists():
                self.best_path.unlink()  # Remove old best
            
            self.best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, self.best_path)
            self.best_score = metric_value
    
    def load(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
             path: Optional[str] = None, load_best: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            path: Path to checkpoint (if None, loads best or last)
            load_best: If True and path is None, load best; else load last
        
        Returns:
            Checkpoint dictionary
        """
        if path is None:
            if load_best:
                path = self.checkpoint_dir / 'checkpoint_best.pth'
            else:
                path = self.checkpoint_dir / 'checkpoint_last.pth'
        
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


class LearningRateScheduler:
    """Learning rate scheduler wrapper."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler_type: str = 'cosine',
                 **scheduler_kwargs):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            scheduler_type: Type of scheduler ('cosine', 'step', 'plateau', 'onecycle')
            **scheduler_kwargs: Additional arguments for scheduler
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_kwargs
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **scheduler_kwargs
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_kwargs
            )
        elif scheduler_type == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, **scheduler_kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric: Optional[float] = None):
        """
        Step the scheduler.
        
        Args:
            metric: Metric value (required for 'plateau' scheduler)
        """
        if self.scheduler_type == 'plateau':
            if metric is None:
                raise ValueError("Metric required for plateau scheduler")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_last_lr(self):
        """Get current learning rate."""
        return self.scheduler.get_last_lr()

