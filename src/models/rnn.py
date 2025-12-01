"""RNN models (LSTM/GRU) with attention for time-series classification."""

import torch
import torch.nn as nn
from typing import Optional


class AttentionLayer(nn.Module):
    """Attention mechanism for RNN outputs."""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, rnn_outputs):
        """
        Apply attention to RNN outputs.
        
        Args:
            rnn_outputs: Tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Weighted sum of RNN outputs
        """
        # Compute attention weights
        attention_weights = self.attention(rnn_outputs)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        attended = torch.sum(attention_weights * rnn_outputs, dim=1)  # (batch_size, hidden_size)
        return attended


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with attention for time-series classification.
    
    Args:
        input_size: Number of input features per time step
        hidden_size: Hidden state size
        num_layers: Number of LSTM layers
        num_classes: Number of output classes
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
        use_attention: Whether to use attention mechanism
        multitask: If True, also output glucose regression
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 multitask: bool = False):
        super(LSTMClassifier, self).__init__()
        
        self.multitask = multitask
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention layer
        if use_attention:
            lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
            self.attention = AttentionLayer(lstm_output_size)
        
        # Classification head
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Linear(lstm_output_size, num_classes)
        
        # Regression head (for multitask)
        if multitask:
            self.regressor = nn.Linear(lstm_output_size, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Classification logits, and optionally regression output
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention or use last hidden state
        if self.use_attention:
            features = self.attention(lstm_out)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                features = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                features = h_n[-1]
        
        # Classification
        logits = self.classifier(features)
        
        if self.multitask:
            glucose_pred = self.regressor(features)
            return logits, glucose_pred
        
        return logits


class GRUClassifier(nn.Module):
    """
    Bidirectional GRU with attention for time-series classification.
    
    Args:
        input_size: Number of input features per time step
        hidden_size: Hidden state size
        num_layers: Number of GRU layers
        num_classes: Number of output classes
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional GRU
        use_attention: Whether to use attention mechanism
        multitask: If True, also output glucose regression
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 multitask: bool = False):
        super(GRUClassifier, self).__init__()
        
        self.multitask = multitask
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention layer
        if use_attention:
            gru_output_size = hidden_size * 2 if bidirectional else hidden_size
            self.attention = AttentionLayer(gru_output_size)
        
        # Classification head
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Linear(gru_output_size, num_classes)
        
        # Regression head (for multitask)
        if multitask:
            self.regressor = nn.Linear(gru_output_size, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Classification logits, and optionally regression output
        """
        # GRU forward pass
        gru_out, h_n = self.gru(x)
        
        # Apply attention or use last hidden state
        if self.use_attention:
            features = self.attention(gru_out)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                features = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                features = h_n[-1]
        
        # Classification
        logits = self.classifier(features)
        
        if self.multitask:
            glucose_pred = self.regressor(features)
            return logits, glucose_pred
        
        return logits

