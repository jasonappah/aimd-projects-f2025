"""Temporal Convolutional Network (TCN) for time-series classification."""

import torch
import torch.nn as nn
from typing import Optional


class TemporalBlock(nn.Module):
    """Temporal block with dilated convolution, normalization, and dropout."""
    
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 padding: int,
                 dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """Forward pass."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for time-series classification.
    
    Args:
        input_size: Number of input features per time step
        num_classes: Number of output classes
        num_channels: List of channel sizes for each layer
        kernel_size: Convolution kernel size
        dropout: Dropout probability
        multitask: If True, also output glucose regression
    """
    
    def __init__(self,
                 input_size: int,
                 num_classes: int = 2,
                 num_channels: list = [64, 128, 256],
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 multitask: bool = False):
        super(TCN, self).__init__()
        
        self.multitask = multitask
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                    stride=1, dilation=dilation_size,
                                    padding=(kernel_size-1) * dilation_size,
                                    dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Linear(num_channels[-1], num_classes)
        
        # Regression head (for multitask)
        if multitask:
            self.regressor = nn.Linear(num_channels[-1], 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Classification logits, and optionally regression output
        """
        # Reshape to (batch_size, input_size, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Pass through TCN layers
        out = self.network(x)
        
        # Take the last time step
        out = out[:, :, -1]
        
        # Classification
        logits = self.classifier(out)
        
        if self.multitask:
            glucose_pred = self.regressor(out)
            return logits, glucose_pred
        
        return logits

