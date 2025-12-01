"""Train baseline models (Logistic Regression, XGBoost, LightGBM, MLP)."""

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
from src.data.dataset import TabularWindowDataset
from src.models.baselines import LogisticRegressionBaseline, XGBoostBaseline, LightGBMBaseline, SimpleMLP
from src.training.trainer import Trainer
from src.training.losses import ClassWeightedBCE, FocalLoss
from src.training.metrics import compute_all_metrics, compute_operational_metrics
from src.utils.device import get_device, get_device_info


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_baselines(config_path: str = "configs/baseline_config.yaml"):
    """Train all baseline models."""
    config = load_config(config_path)
    
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
    train_dataset = TabularWindowDataset(
        df, window_size=config['data']['window_size'],
        person_ids=train_ids
    )
    val_dataset = TabularWindowDataset(
        df, window_size=config['data']['window_size'],
        person_ids=val_ids
    )
    test_dataset = TabularWindowDataset(
        df, window_size=config['data']['window_size'],
        person_ids=test_ids
    )
    
    # Get feature names
    feature_cols = train_dataset.get_feature_names()
    print(f"Number of features: {len(feature_cols)}")
    
    # Prepare data for sklearn/XGBoost models
    X_train = train_dataset.get_numpy_features()
    y_train = train_dataset.get_numpy_targets()
    X_val = val_dataset.get_numpy_features()
    y_val = val_dataset.get_numpy_targets()
    X_test = test_dataset.get_numpy_features()
    y_test = test_dataset.get_numpy_targets()
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    print(f"Positive class ratio - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
    print()
    
    results = {}
    
    # 1. Logistic Regression
    print("=" * 60)
    print("Training Logistic Regression...")
    print("=" * 60)
    lr_model = LogisticRegressionBaseline(**config['models']['logistic_regression'])
    lr_model.fit(X_train, y_train)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    lr_metrics = compute_all_metrics(y_test, lr_proba)
    lr_ops = compute_operational_metrics(y_test, (lr_proba >= lr_metrics['optimal_threshold']).astype(int))
    results['logistic_regression'] = {**lr_metrics, **lr_ops}
    print(f"Test PR-AUC: {lr_metrics['pr_auc']:.4f}, ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    print()
    
    # 2. XGBoost
    print("=" * 60)
    print("Training XGBoost...")
    print("=" * 60)
    xgb_model = XGBoostBaseline(**config['models']['xgboost'])
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_all_metrics(y_test, xgb_proba)
    xgb_ops = compute_operational_metrics(y_test, (xgb_proba >= xgb_metrics['optimal_threshold']).astype(int))
    results['xgboost'] = {**xgb_metrics, **xgb_ops}
    print(f"Test PR-AUC: {xgb_metrics['pr_auc']:.4f}, ROC-AUC: {xgb_metrics['roc_auc']:.4f}")
    print()
    
    # 3. LightGBM
    print("=" * 60)
    print("Training LightGBM...")
    print("=" * 60)
    lgb_model = LightGBMBaseline(**config['models']['lightgbm'])
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    lgb_metrics = compute_all_metrics(y_test, lgb_proba)
    lgb_ops = compute_operational_metrics(y_test, (lgb_proba >= lgb_metrics['optimal_threshold']).astype(int))
    results['lightgbm'] = {**lgb_metrics, **lgb_ops}
    print(f"Test PR-AUC: {lgb_metrics['pr_auc']:.4f}, ROC-AUC: {lgb_metrics['roc_auc']:.4f}")
    print()
    
    # 4. MLP
    print("=" * 60)
    print("Training MLP...")
    print("=" * 60)
    mlp_model = SimpleMLP(
        input_size=len(feature_cols),
        hidden_sizes=config['models']['mlp']['hidden_sizes'],
        dropout=config['models']['mlp']['dropout']
    )
    
    # Calculate class weights
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0], dtype=torch.float32).to(device)
    
    loss_fn = ClassWeightedBCE(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        mlp_model.parameters(),
        lr=config['models']['mlp']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['models']['mlp']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['models']['mlp']['batch_size'], shuffle=False)
    
    trainer = Trainer(mlp_model, device=device, loss_fn=loss_fn, optimizer=optimizer)
    history = trainer.fit(train_loader, val_loader, num_epochs=config['models']['mlp']['num_epochs'])
    
    # Evaluate on test set
    test_loader = DataLoader(test_dataset, batch_size=config['models']['mlp']['batch_size'], shuffle=False)
    mlp_metrics = trainer.validate(test_loader)
    
    # Get predictions for operational metrics
    mlp_model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            logits = mlp_model(features)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
    
    all_preds = np.array(all_preds)
    mlp_preds = (all_preds >= mlp_metrics['optimal_threshold']).astype(int)
    mlp_ops = compute_operational_metrics(y_test, mlp_preds)
    results['mlp'] = {**mlp_metrics, **mlp_ops}
    print(f"Test PR-AUC: {mlp_metrics['pr_auc']:.4f}, ROC-AUC: {mlp_metrics['roc_auc']:.4f}")
    print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  F1: {metrics['f1_at_threshold']:.4f}")
        print(f"  FN per 10k hours: {metrics.get('false_negatives_per_10k_hours', 0):.2f}")
        print()
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml",
                       help="Path to config file")
    args = parser.parse_args()
    
    results = train_baselines(args.config)

