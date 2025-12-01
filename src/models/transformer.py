"""Transformer-based model for time-series classification."""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time-series classification.
    
    Args:
        input_size: Number of input features per time step
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Feedforward dimension
        num_classes: Number of output classes
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for positional encoding
        multitask: If True, also output glucose regression
    """
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 max_seq_len: int = 100,
                 multitask: bool = False):
        super(TimeSeriesTransformer, self).__init__()
        
        self.multitask = multitask
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Regression head (for multitask)
        if multitask:
            self.regressor = nn.Linear(d_model, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Classification logits, and optionally regression output
        """
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        # Note: Transformer expects (seq_len, batch_size, d_model) by default,
        # but with batch_first=True, we use (batch_size, seq_len, d_model)
        encoded = self.transformer_encoder(x)
        
        # Use mean pooling over sequence length (or could use CLS token)
        # Alternative: use last time step
        features = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(features)
        
        if self.multitask:
            glucose_pred = self.regressor(features)
            return logits, glucose_pred
        
        return logits

