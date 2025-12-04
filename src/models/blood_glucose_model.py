"""
Deep neural network model for blood glucose prediction.
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class BloodGlucoseModel(nn.Module):
    """
    Deep feedforward neural network for predicting next 3-hour glucose levels.
    
    Architecture:
    - Input layer
    - 4 hidden layers with decreasing width (256, 128, 64, 32)
    - Batch normalization after each hidden layer
    - Dropout for regularization
    - ReLU activations
    - Single output neuron for regression
    """
    
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], 
                 dropout_rate=0.4, activation='relu'):
        """
        Initialize the model.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes (default: [256, 128, 64, 32])
            dropout_rate: Dropout probability (default: 0.4)
            activation: Activation function ('relu' or 'gelu')
        """
        super(BloodGlucoseModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # Hidden layers with batch norm and dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer (single neuron for regression)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Kaiming initialization for ReLU
                init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.network(x)
    
    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

