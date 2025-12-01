"""Baseline models: Logistic Regression, XGBoost, LightGBM, MLP."""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from typing import Optional, Dict, Any
import numpy as np


class LogisticRegressionBaseline:
    """Logistic Regression baseline using sklearn."""
    
    def __init__(self, class_weight: Optional[str] = 'balanced', **kwargs):
        self.model = LogisticRegression(class_weight=class_weight, **kwargs)
    
    def fit(self, X, y):
        """Train the model."""
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict classes."""
        return self.model.predict(X)


class XGBoostBaseline:
    """XGBoost baseline."""
    
    def __init__(self, 
                 scale_pos_weight: Optional[float] = None,
                 eval_metric: str = 'aucpr',
                 **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            scale_pos_weight: Weight for positive class (None = auto from data)
            eval_metric: Evaluation metric
            **kwargs: Additional XGBoost parameters
        """
        self.scale_pos_weight = scale_pos_weight
        self.eval_metric = eval_metric
        self.kwargs = kwargs
        self.model = None
    
    def fit(self, X, y, eval_set=None, **fit_kwargs):
        """Train the model."""
        # Auto-calculate scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            if n_pos > 0 and n_neg > 0:
                self.scale_pos_weight = n_neg / n_pos
            else:
                self.scale_pos_weight = 1.0
        
        params = {
            'scale_pos_weight': self.scale_pos_weight,
            'eval_metric': self.eval_metric,
            'objective': 'binary:logistic',
            **self.kwargs
        }
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y, eval_set=eval_set, **fit_kwargs)
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict classes."""
        return self.model.predict(X)


class LightGBMBaseline:
    """LightGBM baseline."""
    
    def __init__(self,
                 scale_pos_weight: Optional[float] = None,
                 **kwargs):
        """
        Initialize LightGBM model.
        
        Args:
            scale_pos_weight: Weight for positive class
            **kwargs: Additional LightGBM parameters
        """
        self.scale_pos_weight = scale_pos_weight
        self.kwargs = kwargs
        self.model = None
    
    def fit(self, X, y, eval_set=None, **fit_kwargs):
        """Train the model."""
        # Auto-calculate scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            if n_pos > 0 and n_neg > 0:
                self.scale_pos_weight = n_neg / n_pos
            else:
                self.scale_pos_weight = 1.0
        
        params = {
            'scale_pos_weight': self.scale_pos_weight,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            **self.kwargs
        }
        
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X, y, eval_set=eval_set, **fit_kwargs)
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict classes."""
        return self.model.predict(X)


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for tabular data."""
    
    def __init__(self,
                 input_size: int,
                 hidden_sizes: list = [128, 64],
                 dropout: float = 0.2,
                 num_classes: int = 2,
                 multitask: bool = False):
        """
        Initialize MLP.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
            num_classes: Number of output classes
            multitask: If True, also output glucose regression
        """
        super(SimpleMLP, self).__init__()
        
        self.multitask = multitask
        layers = []
        
        # Input layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Linear(prev_size, num_classes)
        
        # Regression head (for multitask)
        if multitask:
            self.regressor = nn.Linear(prev_size, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Classification logits, and optionally regression output
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        
        if self.multitask:
            glucose_pred = self.regressor(features)
            return logits, glucose_pred
        
        return logits

