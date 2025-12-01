"""Device detection and configuration for PyTorch backends (ROCm, CUDA, MPS, CPU)."""

import torch
import warnings
from typing import Optional


def get_device(device_override: Optional[str] = None) -> torch.device:
    """
    Auto-detect the best available PyTorch backend.
    
    Priority: ROCm → CUDA → MPS → CPU
    
    Args:
        device_override: Optional device string to override auto-detection
                        (e.g., 'cuda', 'mps', 'cpu')
    
    Returns:
        torch.device: The selected device
    """
    if device_override:
        device = torch.device(device_override)
        if device.type == 'cuda' and not torch.cuda.is_available():
            warnings.warn(f"CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        elif device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            warnings.warn(f"MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
        return device
    
    # Check for ROCm (uses CUDA API but has 'hip' in version)
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        if torch.cuda.is_available():
            return torch.device("cuda")
    
    # Check for CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    
    # Fallback to CPU
    return torch.device("cpu")


def get_device_info(device: torch.device) -> dict:
    """
    Get information about the device.
    
    Args:
        device: The torch device
    
    Returns:
        dict: Device information
    """
    info = {
        "device": str(device),
        "type": device.type,
    }
    
    if device.type == 'cuda':
        info["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["device_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
        if hasattr(torch.version, 'hip'):
            info["rocm_version"] = torch.version.hip
    elif device.type == 'mps':
        info["mps_available"] = torch.backends.mps.is_available()
        info["mps_built"] = torch.backends.mps.is_built()
    
    return info


def to_device(tensor, device: torch.device, non_blocking: bool = False):
    """
    Move tensor to device (device-agnostic helper).
    
    Args:
        tensor: Tensor or tensor-like object
        device: Target device
        non_blocking: Whether to use non-blocking transfer (CUDA only)
    
    Returns:
        Tensor on the target device
    """
    if hasattr(tensor, 'to'):
        return tensor.to(device, non_blocking=non_blocking)
    return tensor

