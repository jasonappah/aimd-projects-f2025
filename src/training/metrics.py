"""Evaluation metrics for model performance."""

import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix
)
from typing import Tuple, Dict, Optional


def compute_pr_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute Precision-Recall AUC.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
    
    Returns:
        PR-AUC score
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)


def compute_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute ROC AUC.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
    
    Returns:
        ROC-AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return auc(fpr, tpr)


def compute_f1_at_threshold(y_true: np.ndarray, 
                            y_pred_proba: np.ndarray,
                            threshold: float = 0.5) -> float:
    """
    Compute F1 score at a given threshold.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        threshold: Classification threshold
    
    Returns:
        F1 score
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return f1_score(y_true, y_pred)


def compute_optimal_threshold(y_true: np.ndarray,
                             y_pred_proba: np.ndarray,
                             metric: str = 'f1') -> Tuple[float, float]:
    """
    Find optimal threshold for a given metric.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        metric: Metric to optimize ('f1', 'precision', 'recall')
    
    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    if metric == 'f1':
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx], f1_scores[optimal_idx]
    elif metric == 'precision':
        optimal_idx = np.argmax(precision)
        return thresholds[optimal_idx], precision[optimal_idx]
    elif metric == 'recall':
        optimal_idx = np.argmax(recall)
        return thresholds[optimal_idx], recall[optimal_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute Brier score (calibration metric).
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
    
    Returns:
        Brier score (lower is better)
    """
    return brier_score_loss(y_true, y_pred_proba)


def compute_confusion_matrix_metrics(y_true: np.ndarray,
                                     y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics from confusion matrix.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
    
    Returns:
        Dictionary of metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1
    }


def compute_all_metrics(y_true: np.ndarray,
                       y_pred_proba: np.ndarray,
                       threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        threshold: Classification threshold (if None, find optimal)
    
    Returns:
        Dictionary of all metrics
    """
    # Find optimal threshold if not provided
    if threshold is None:
        threshold, optimal_f1 = compute_optimal_threshold(y_true, y_pred_proba, 'f1')
    else:
        optimal_f1 = compute_f1_at_threshold(y_true, y_pred_proba, threshold)
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'pr_auc': compute_pr_auc(y_true, y_pred_proba),
        'roc_auc': compute_roc_auc(y_true, y_pred_proba),
        'brier_score': compute_brier_score(y_true, y_pred_proba),
        'optimal_threshold': threshold,
        'f1_at_threshold': optimal_f1,
    }
    
    # Add confusion matrix metrics
    cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)
    metrics.update(cm_metrics)
    
    return metrics


def compute_operational_metrics(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                hours_per_sample: float = 1.0) -> Dict[str, float]:
    """
    Compute operational metrics (false negatives per 10k hours, etc.).
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        hours_per_sample: Number of hours per sample (default 1 for hourly data)
    
    Returns:
        Dictionary of operational metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_hours = len(y_true) * hours_per_sample
    total_samples = len(y_true)
    
    # False negatives per 10k hours
    fn_per_10k_hours = (fn / total_hours) * 10000 if total_hours > 0 else 0
    
    # False alarms per day (assuming hourly data, 24 hours per day)
    hours_per_day = 24
    false_alarms_per_day = (fp / total_hours) * hours_per_day if total_hours > 0 else 0
    
    return {
        'false_negatives_per_10k_hours': fn_per_10k_hours,
        'false_alarms_per_day': false_alarms_per_day,
        'total_hours': total_hours,
        'total_samples': total_samples
    }

