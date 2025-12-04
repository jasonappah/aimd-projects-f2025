"""
Device detection utility for PyTorch.
Automatically detects and returns the best available device (ROCm/CUDA/MPS/CPU).
"""

import torch
import os


def get_device():
    """
    Detect and return the best available PyTorch device.
    
    Priority order:
    1. CUDA (NVIDIA GPU)
    2. ROCm (AMD GPU)
    3. MPS (Apple Silicon GPU)
    4. CPU (fallback)
    
    Returns:
        torch.device: The detected device
    """
    # Check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    
    # Check for ROCm (AMD)
    # ROCm uses HIP backend, which can be detected via environment or torch.version
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        # Try to use ROCm if available
        try:
            if torch.cuda.is_available():  # ROCm also uses cuda.is_available()
                device = torch.device("cuda")
                print(f"Using ROCm device (HIP backend)")
                return device
        except:
            pass
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
        return device
    
    # Fallback to CPU
    device = torch.device("cpu")
    print("Using CPU device")
    return device


def get_device_dtype():
    """
    Get the appropriate dtype for the current device.
    MPS has some limitations with float64, so we use float32.
    
    Returns:
        torch.dtype: The appropriate dtype for the device
    """
    device = get_device()
    if device.type == "mps":
        # MPS works best with float32
        return torch.float32
    else:
        return torch.float32  # float32 is standard for most training
