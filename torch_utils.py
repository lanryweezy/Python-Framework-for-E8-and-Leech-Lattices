"""
Torch utilities for E8 Leech Lattice Framework.

This module provides utilities for integrating with PyTorch, including
TorchScript support for model serialization and deployment.
"""

import torch
import numpy as np
from typing import Any, Dict, Union
from .logging_utils import get_logger

logger = get_logger(__name__)

def export_to_torchscript(model: torch.nn.Module, filepath: str) -> None:
    """
    Export a PyTorch model to TorchScript format.
    
    Args:
        model: The PyTorch model to export.
        filepath: The path to save the TorchScript model.
    """
    try:
        # Trace the model
        # For models with control flow or dynamic shapes, scripting might be needed
        # Here, we assume a simple traceable model for demonstration
        example_input = torch.randn(1, model.input_dim) if hasattr(model, 'input_dim') else torch.randn(1, 10) # Placeholder
        traced_model = torch.jit.trace(model, example_input)
        
        # Save the traced model
        traced_model.save(filepath)
        logger.info(f"Model successfully exported to TorchScript: {filepath}")
    except Exception as e:
        logger.error(f"Error exporting model to TorchScript: {e}")
        raise

def load_from_torchscript(filepath: str) -> torch.jit.ScriptModule:
    """
    Load a TorchScript model from a file.
    
    Args:
        filepath: The path to the TorchScript model file.
        
    Returns:
        The loaded TorchScript model.
    """
    try:
        model = torch.jit.load(filepath)
        logger.info(f"Model successfully loaded from TorchScript: {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from TorchScript: {e}")
        raise

def convert_to_torch_tensor(data: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert various data types (numpy array, list) to a PyTorch tensor.
    
    Args:
        data: The data to convert.
        dtype: The desired data type for the tensor.
        
    Returns:
        A PyTorch tensor.
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype)
    elif isinstance(data, list):
        return torch.tensor(data, dtype=dtype)
    elif isinstance(data, torch.Tensor):
        return data.to(dtype)
    else:
        raise TypeError(f"Unsupported data type for conversion to tensor: {type(data)}")

def convert_to_numpy_array(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.
    
    Args:
        tensor: The PyTorch tensor to convert.
        
    Returns:
        A NumPy array.
    """
    if tensor.is_cuda:
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()


# Example usage and testing
if __name__ == "__main__":
    class SimpleNN(torch.nn.Module):
        def __init__(self, input_dim: int = 10, output_dim: int = 1):
            super().__init__()
            self.input_dim = input_dim
            self.linear = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    # Create a dummy model
    model = SimpleNN(input_dim=5, output_dim=1)
    dummy_input = torch.randn(1, 5)

    # Test export to TorchScript
    torchscript_filepath = "simple_nn.pt"
    try:
        # Trace the model with a dummy input
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(torchscript_filepath)
        print(f"Model successfully exported to TorchScript: {torchscript_filepath}")

        # Test load from TorchScript
        loaded_model = load_from_torchscript(torchscript_filepath)
        output = loaded_model(dummy_input)
        print(f"Loaded model output: {output}")
        print("TorchScript export/load successful!")
    except Exception as e:
        print(f"TorchScript test failed: {e}")

    # Test tensor conversions
    print("\n--- Testing Tensor Conversions ---")
    numpy_array = np.array([1.0, 2.0, 3.0])
    list_data = [4.0, 5.0, 6.0]
    
    tensor_from_numpy = convert_to_torch_tensor(numpy_array)
    print(f"Tensor from NumPy: {tensor_from_numpy}")
    assert isinstance(tensor_from_numpy, torch.Tensor)
    
    tensor_from_list = convert_to_torch_tensor(list_data)
    print(f"Tensor from List: {tensor_from_list}")
    assert isinstance(tensor_from_list, torch.Tensor)
    
    numpy_from_tensor = convert_to_numpy_array(tensor_from_numpy)
    print(f"NumPy from Tensor: {numpy_from_tensor}")
    assert isinstance(numpy_from_tensor, np.ndarray)
    
    print("Tensor conversions successful!")

    print("All Torch utilities tests passed!")

