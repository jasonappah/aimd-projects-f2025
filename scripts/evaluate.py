"""Evaluate trained models."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import argparse

from src.training.callbacks import ModelCheckpoint
from src.training.metrics import compute_all_metrics, compute_operational_metrics
from src.utils.device import get_device


def evaluate_model(model, checkpoint_path: str, test_loader, device):
    """Evaluate a trained model."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            # Handle multitask models
            if isinstance(model_output := model(features), tuple):
                logits, _ = model_output
            else:
                logits = model_output
            
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    metrics = compute_all_metrics(all_targets, all_preds)
    predictions = (all_preds >= metrics['optimal_threshold']).astype(int)
    ops_metrics = compute_operational_metrics(all_targets, predictions)
    
    return {**metrics, **ops_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, required=True,
                      help="Path to test dataset")
    args = parser.parse_args()
    
    # This is a template - would need to load model and test_loader based on model type
    print("Evaluation script - implement based on model type")

