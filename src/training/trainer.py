"""Main training loop for PyTorch models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Callable
import warnings

from src.training.metrics import compute_all_metrics
from src.training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from src.utils.device import get_device, to_device


class Trainer:
    """Trainer class for PyTorch models."""
    
    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 loss_fn: Optional[nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 multitask: bool = False):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on (auto-detected if None)
            loss_fn: Loss function
            optimizer: Optimizer
            multitask: Whether model is multitask (classification + regression)
        """
        self.model = model
        self.device = device if device is not None else get_device()
        self.model.to(self.device)
        self.multitask = multitask
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1} [Train]"):
            features = to_device(batch['features'], self.device)
            targets = to_device(batch['target'], self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.multitask:
                logits, glucose_pred = self.model(features)
                glucose_targets = to_device(batch['glucose_target'], self.device)
                
                # Compute loss (assuming multitask loss)
                if isinstance(self.loss_fn, nn.Module):
                    loss, cls_loss, reg_loss = self.loss_fn(
                        logits, glucose_pred, targets, glucose_targets
                    )
                else:
                    # Fallback to separate losses
                    cls_loss = nn.CrossEntropyLoss()(logits, targets)
                    reg_loss = nn.MSELoss()(glucose_pred.squeeze(), glucose_targets.squeeze())
                    loss = cls_loss + 0.1 * reg_loss
            else:
                logits = self.model(features)
                loss = self.loss_fn(logits, targets) if self.loss_fn else nn.CrossEntropyLoss()(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        metrics = compute_all_metrics(np.array(all_targets), np.array(all_preds))
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            dataloader: Validation dataloader
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1} [Val]"):
                features = to_device(batch['features'], self.device)
                targets = to_device(batch['target'], self.device)
                
                if self.multitask:
                    logits, glucose_pred = self.model(features)
                    glucose_targets = to_device(batch['glucose_target'], self.device)
                    
                    if isinstance(self.loss_fn, nn.Module):
                        loss, cls_loss, reg_loss = self.loss_fn(
                            logits, glucose_pred, targets, glucose_targets
                        )
                    else:
                        cls_loss = nn.CrossEntropyLoss()(logits, targets)
                        reg_loss = nn.MSELoss()(glucose_pred.squeeze(), glucose_targets.squeeze())
                        loss = cls_loss + 0.1 * reg_loss
                else:
                    logits = self.model(features)
                    loss = self.loss_fn(logits, targets) if self.loss_fn else nn.CrossEntropyLoss()(logits, targets)
                
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        metrics = compute_all_metrics(np.array(all_targets), np.array(all_preds))
        metrics['loss'] = avg_loss
        
        return metrics
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 50,
            early_stopping: Optional[EarlyStopping] = None,
            checkpoint: Optional[ModelCheckpoint] = None,
            lr_scheduler: Optional[LearningRateScheduler] = None,
            verbose: bool = True) -> Dict[str, list]:
        """
        Train model.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs
            early_stopping: Early stopping callback
            checkpoint: Model checkpoint callback
            lr_scheduler: Learning rate scheduler
            verbose: Whether to print progress
        
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_pr_auc': [],
            'val_roc_auc': [],
            'val_f1': []
        }
        
        best_val_metric = None
        best_val_pr_auc = 0.0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_pr_auc'].append(val_metrics['pr_auc'])
            history['val_roc_auc'].append(val_metrics['roc_auc'])
            history['val_f1'].append(val_metrics['f1_at_threshold'])
            
            # Check if best model
            is_best = val_metrics['pr_auc'] > best_val_pr_auc
            if is_best:
                best_val_pr_auc = val_metrics['pr_auc']
                best_val_metric = val_metrics
            
            # Save checkpoint
            if checkpoint:
                checkpoint.save(
                    self.model, self.optimizer, epoch, val_metrics['pr_auc'],
                    val_metrics, is_best=is_best
                )
            
            # Learning rate scheduling
            if lr_scheduler:
                if lr_scheduler.scheduler_type == 'plateau':
                    lr_scheduler.step(val_metrics['pr_auc'])
                else:
                    lr_scheduler.step()
            
            # Print progress
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val PR-AUC: {val_metrics['pr_auc']:.4f}")
                print(f"  Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
                print(f"  Val F1: {val_metrics['f1_at_threshold']:.4f}")
                if lr_scheduler:
                    print(f"  LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                print()
            
            # Early stopping
            if early_stopping:
                if early_stopping(val_metrics['pr_auc']):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return history

