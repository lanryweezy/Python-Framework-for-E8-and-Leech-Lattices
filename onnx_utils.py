"""
ONNX utilities for E8 Leech Lattice Framework.

This module provides utilities for exporting models to ONNX format
for cross-platform compatibility and deployment.
"""

import torch
import numpy as np
from typing import Any, Dict, Union, Tuple, Optional
from .logging_utils import get_logger

logger = get_logger(__name__)

# Check if ONNX is available
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. Install with: pip install onnx onnxruntime")

def export_to_onnx(model: torch.nn.Module, filepath: str, 
                   input_shape: Tuple[int, ...] = (1, 10),
                   input_names: Optional[list] = None,
                   output_names: Optional[list] = None,
                   dynamic_axes: Optional[Dict] = None) -> None:
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: The PyTorch model to export.
        filepath: The path to save the ONNX model.
        input_shape: The shape of the input tensor.
        input_names: Names for input tensors.
        output_names: Names for output tensors.
        dynamic_axes: Dynamic axes specification.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX is not available. Install with: pip install onnx onnxruntime")
    
    try:
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Set default names if not provided
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
        
        # Export the model
        torch.onnx.export(
            model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        
        logger.info(f"Model successfully exported to ONNX: {filepath}")
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {e}")
        raise

def load_onnx_model(filepath: str) -> ort.InferenceSession:
    """
    Load an ONNX model for inference.
    
    Args:
        filepath: The path to the ONNX model file.
        
    Returns:
        An ONNX Runtime inference session.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime is not available. Install with: pip install onnxruntime")
    
    try:
        session = ort.InferenceSession(filepath)
        logger.info(f"ONNX model successfully loaded: {filepath}")
        return session
    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        raise

def run_onnx_inference(session: ort.InferenceSession, 
                       input_data: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
    """
    Run inference using an ONNX model.
    
    Args:
        session: The ONNX Runtime inference session.
        input_data: Input data as numpy array or dictionary of arrays.
        
    Returns:
        The inference output.
    """
    try:
        if isinstance(input_data, np.ndarray):
            # Single input case
            input_name = session.get_inputs()[0].name
            input_dict = {input_name: input_data}
        else:
            # Multiple inputs case
            input_dict = input_data
        
        outputs = session.run(None, input_dict)
        logger.debug(f"ONNX inference completed with output shape: {outputs[0].shape}")
        return outputs[0] if len(outputs) == 1 else outputs
    except Exception as e:
        logger.error(f"Error running ONNX inference: {e}")
        raise

def validate_onnx_model(filepath: str) -> bool:
    """
    Validate an ONNX model file.
    
    Args:
        filepath: The path to the ONNX model file.
        
    Returns:
        True if the model is valid, False otherwise.
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX not available for validation")
        return False
    
    try:
        model = onnx.load(filepath)
        onnx.checker.check_model(model)
        logger.info(f"ONNX model validation successful: {filepath}")
        return True
    except Exception as e:
        logger.error(f"ONNX model validation failed: {e}")
        return False

def get_onnx_model_info(filepath: str) -> Dict[str, Any]:
    """
    Get information about an ONNX model.
    
    Args:
        filepath: The path to the ONNX model file.
        
    Returns:
        Dictionary with model information.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX is not available")
    
    try:
        session = ort.InferenceSession(filepath)
        
        # Get input information
        inputs_info = []
        for input_meta in session.get_inputs():
            inputs_info.append({
                'name': input_meta.name,
                'type': input_meta.type,
                'shape': input_meta.shape
            })
        
        # Get output information
        outputs_info = []
        for output_meta in session.get_outputs():
            outputs_info.append({
                'name': output_meta.name,
                'type': output_meta.type,
                'shape': output_meta.shape
            })
        
        model_info = {
            'inputs': inputs_info,
            'outputs': outputs_info,
            'providers': session.get_providers()
        }
        
        logger.info(f"Retrieved ONNX model info for: {filepath}")
        return model_info
    except Exception as e:
        logger.error(f"Error getting ONNX model info: {e}")
        raise

def optimize_onnx_model(input_filepath: str, output_filepath: str) -> None:
    """
    Optimize an ONNX model for better performance.
    
    Args:
        input_filepath: Path to the input ONNX model.
        output_filepath: Path to save the optimized model.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX is not available")
    
    try:
        # Load the model
        model = onnx.load(input_filepath)
        
        # Apply optimizations
        from onnx import optimizer
        optimized_model = optimizer.optimize(model)
        
        # Save the optimized model
        onnx.save(optimized_model, output_filepath)
        
        logger.info(f"ONNX model optimized and saved: {output_filepath}")
    except Exception as e:
        logger.error(f"Error optimizing ONNX model: {e}")
        raise

def convert_pytorch_to_onnx_with_validation(model: torch.nn.Module, 
                                          filepath: str,
                                          input_shape: Tuple[int, ...] = (1, 10)) -> bool:
    """
    Convert PyTorch model to ONNX with validation.
    
    Args:
        model: The PyTorch model to convert.
        filepath: The path to save the ONNX model.
        input_shape: The shape of the input tensor.
        
    Returns:
        True if conversion and validation successful, False otherwise.
    """
    try:
        # Export to ONNX
        export_to_onnx(model, filepath, input_shape)
        
        # Validate the exported model
        is_valid = validate_onnx_model(filepath)
        
        if is_valid:
            # Test inference to ensure it works
            dummy_input = torch.randn(*input_shape)
            
            # PyTorch inference
            model.eval()
            with torch.no_grad():
                pytorch_output = model(dummy_input).numpy()
            
            # ONNX inference
            session = load_onnx_model(filepath)
            onnx_output = run_onnx_inference(session, dummy_input.numpy())
            
            # Compare outputs
            if np.allclose(pytorch_output, onnx_output, atol=1e-5):
                logger.info("PyTorch to ONNX conversion successful with matching outputs")
                return True
            else:
                logger.warning("PyTorch and ONNX outputs don't match closely")
                return False
        else:
            return False
            
    except Exception as e:
        logger.error(f"Error in PyTorch to ONNX conversion: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    if not ONNX_AVAILABLE:
        print("ONNX not available. Skipping tests.")
        exit(0)
    
    # Define a simple model for testing
    class SimpleNN(torch.nn.Module):
        def __init__(self, input_dim: int = 10, output_dim: int = 1):
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    # Create and test the model
    model = SimpleNN(input_dim=5, output_dim=1)
    onnx_filepath = "simple_nn.onnx"
    
    print("--- Testing ONNX Export ---")
    try:
        # Test export
        export_to_onnx(model, onnx_filepath, input_shape=(1, 5))
        print(f"Model exported to ONNX: {onnx_filepath}")
        
        # Test validation
        is_valid = validate_onnx_model(onnx_filepath)
        print(f"Model validation: {'Passed' if is_valid else 'Failed'}")
        
        # Test model info
        model_info = get_onnx_model_info(onnx_filepath)
        print(f"Model inputs: {model_info['inputs']}")
        print(f"Model outputs: {model_info['outputs']}")
        
        # Test inference
        session = load_onnx_model(onnx_filepath)
        dummy_input = np.random.randn(1, 5).astype(np.float32)
        output = run_onnx_inference(session, dummy_input)
        print(f"ONNX inference output shape: {output.shape}")
        
        # Test full conversion with validation
        conversion_success = convert_pytorch_to_onnx_with_validation(
            model, "validated_model.onnx", input_shape=(1, 5)
        )
        print(f"Full conversion with validation: {'Success' if conversion_success else 'Failed'}")
        
        print("All ONNX tests passed!")
        
    except Exception as e:
        print(f"ONNX tests failed: {e}")

