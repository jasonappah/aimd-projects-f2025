"""Train deep learning models (TCN, LSTM, Transformer)."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.preprocessing import load_data, handle_missing_values, create_person_splits, normalize_features
from src.data.features import create_all_features
from src.data.dataset import GlucoseTimeSeriesDataset
from src.models.tcn import TCN
from src.models.rnn import LSTMClassifier, GRUClassifier
from src.models.transformer import TimeSeriesTransformer
from src.training.trainer import Trainer
from src.training.losses import FocalLoss, ClassWeightedBCE, MultitaskLoss
from src.training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from src.training.metrics import compute_all_metrics, compute_operational_metrics
from src.utils.device import get_device, get_device_info


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(model_type: str, input_size: int, config: dict) -> nn.Module:
    """Create model based on type."""
    model_config = config['model']
    
    if model_type == 'tcn':
        return TCN(
            input_size=input_size,
            num_classes=model_config['num_classes'],
            num_channels=model_config['num_channels'],
            kernel_size=model_config['kernel_size'],
            dropout=model_config['dropout'],
            multitask=model_config['multitask']
        )
    elif model_type == 'lstm':
        return LSTMClassifier(
            input_size=input_size,
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            bidirectional=model_config['bidirectional'],
            use_attention=model_config['use_attention'],
            multitask=model_config['multitask']
        )
    elif model_type == 'gru':
        return GRUClassifier(
            input_size=input_size,
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            bidirectional=model_config['bidirectional'],
            use_attention=model_config['use_attention'],
            multitask=model_config['multitask']
        )
    elif model_type == 'transformer':
        return TimeSeriesTransformer(
            input_size=input_size,
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            max_seq_len=model_config['max_seq_len'],
            multitask=model_config['multitask']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_loss_fn(config: dict, y_train: np.ndarray, device: torch.device) -> nn.Module:
    """Create loss function based on config."""
    loss_type = config['training']['loss']
    multitask = config['model']['multitask']
    
    if multitask:
        # Multitask loss
        if loss_type == 'focal':
            cls_loss = FocalLoss(
                alpha=config['training']['focal_alpha'],
                gamma=config['training']['focal_gamma']
            )
        else:
            pos_weight = config['training'].get('class_weight_pos')
            if pos_weight is None:
                n_pos = np.sum(y_train == 1)
                n_neg = np.sum(y_train == 0)
                pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0]).to(device)
            else:
                pos_weight = torch.tensor([pos_weight]).to(device)
            cls_loss = ClassWeightedBCE(pos_weight=pos_weight)
        
        reg_loss = nn.MSELoss()
        return MultitaskLoss(
            cls_loss, reg_loss,
            classification_weight=1.0,
            regression_weight=0.1
        )
    else:
        # Single task
        if loss_type == 'focal':
            return FocalLoss(
                alpha=config['training']['focal_alpha'],
                gamma=config['training']['focal_gamma']
            )
        else:
            pos_weight = config['training'].get('class_weight_pos')
            if pos_weight is None:
                n_pos = np.sum(y_train == 1)
                n_neg = np.sum(y_train == 0)
                pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0]).to(device)
            else:
                pos_weight = torch.tensor([pos_weight]).to(device)
            return ClassWeightedBCE(pos_weight=pos_weight)


def train_deep(model_type: str, config_path: str):
    """Train a deep learning model."""
    config = load_config(config_path)
    
    # Print device info
    device = get_device(config.get('device', {}).get('override'))
    print("Device Info:")
    print(get_device_info(device))
    print()
    
    # Load and preprocess data
    print("Loading data...")
    df = load_data(config['data']['csv_path'])
    df = handle_missing_values(
        df,
        forward_fill=config['preprocessing']['forward_fill'],
        add_missing_indicators=config['preprocessing']['add_missing_indicators']
    )
    
    # Create features
    print("Creating features...")
    df = create_all_features(
        df,
        rolling_windows=config['features']['rolling_windows'],
        rolling_cols=config['features']['rolling_cols']
    )
    
    # Create person splits
    train_ids, val_ids, test_ids = create_person_splits(
        df,
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    print(f"Train: {len(train_ids)} people, Val: {len(val_ids)} people, Test: {len(test_ids)} people")
    
    # Normalize features
    numeric_cols = [col for col in df.columns 
                   if col not in ['id', 'timestamp', 'hour', 'hypo_next_3_hours', 
                                'hypo_flag', 'risk_score', 'prick_test', 'missing_measurement',
                                'activity_event', 'gender', 'exercise_level'] and
                   df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    df, scalers = normalize_features(
        df, train_ids, numeric_cols,
        per_person=config['preprocessing']['normalize_per_person']
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = GlucoseTimeSeriesDataset(
        df, window_size=config['data']['window_size'],
        multitask=config['model']['multitask'],
        person_ids=train_ids
    )
    val_dataset = GlucoseTimeSeriesDataset(
        df, window_size=config['data']['window_size'],
        multitask=config['model']['multitask'],
        person_ids=val_ids
    )
    test_dataset = GlucoseTimeSeriesDataset(
        df, window_size=config['data']['window_size'],
        multitask=config['model']['multitask'],
        person_ids=test_ids
    )
    
    # Get input size
    feature_cols = train_dataset.get_feature_names()
    input_size = len(feature_cols)
    print(f"Number of features: {input_size}")
    print(f"Window size: {config['data']['window_size']} hours")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # Get class distribution
    train_targets = np.array([train_dataset[i]['target'].item() for i in range(len(train_dataset))])
    print(f"Positive class ratio - Train: {train_targets.mean():.3f}")
    print()
    
    # Create model
    print(f"Creating {model_type.upper()} model...")
    if config['model']['input_size'] is not None:
        input_size = config['model']['input_size']
    model = create_model(model_type, input_size, config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create loss function
    loss_fn = create_loss_fn(config, train_targets, device)
    
    # Create optimizer
    optimizer_type = config['training']['optimizer'].lower()
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Create callbacks
    early_stopping = EarlyStopping(**config['callbacks']['early_stopping'])
    checkpoint = ModelCheckpoint(**config['callbacks']['checkpoint'])
    
    lr_scheduler = None
    if 'scheduler' in config:
        scheduler_config = config['scheduler'].copy()
        scheduler_type = scheduler_config.pop('type')
        lr_scheduler = LearningRateScheduler(optimizer, scheduler_type, **scheduler_config)
    
    # Create trainer
    trainer = Trainer(
        model, device=device, loss_fn=loss_fn, optimizer=optimizer,
        multitask=config['model']['multitask']
    )
    
    # Train
    print("Starting training...")
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=config['training']['num_epochs'],
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        lr_scheduler=lr_scheduler
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    
    # Get predictions for operational metrics
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            if config['model']['multitask']:
                logits, _ = model(features)
            else:
                logits = model(features)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_preds = (all_preds >= test_metrics['optimal_threshold']).astype(int)
    test_ops = compute_operational_metrics(all_targets, test_preds)
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"F1 Score: {test_metrics['f1_at_threshold']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"Brier Score: {test_metrics['brier_score']:.4f}")
    print(f"False Negatives per 10k hours: {test_ops['false_negatives_per_10k_hours']:.2f}")
    print(f"False Alarms per day: {test_ops['false_alarms_per_day']:.2f}")
    print()
    
    return {
        'model_type': model_type,
        'test_metrics': {**test_metrics, **test_ops},
        'history': history
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['tcn', 'lstm', 'gru', 'transformer'],
                       help="Model type to train")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (default: configs/{model}_config.yaml)")
    args = parser.parse_args()
    
    if args.config is None:
        args.config = f"configs/{args.model}_config.yaml"
    
    results = train_deep(args.model, args.config)

