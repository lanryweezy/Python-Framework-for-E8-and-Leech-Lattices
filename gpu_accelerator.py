"""
GPU Acceleration module using CuPy
"""

import numpy as np
from typing import Union, Tuple, Dict, Any, Optional

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class GPUAccelerator:
    """
    Provides utilities for GPU acceleration using CuPy.
    """
    
    @staticmethod
    def to_gpu(data: np.ndarray):
        """
        Transfers a NumPy array to a CuPy array on the GPU.
        
        Args:
            data: NumPy array to transfer.
            
        Returns:
            CuPy array on the GPU if available, else original array.
        """
        if not CUPY_AVAILABLE:
            logger.warning("CuPy not available, returning original array")
            return data
            
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        logger.info(f"Transferring data to GPU: {data.shape} {data.dtype}")
        return cp.asarray(data)
    
    @staticmethod
    def to_cpu(data):
        """
        Transfers a CuPy array from the GPU to a NumPy array on the CPU.
        
        Args:
            data: CuPy array to transfer.
            
        Returns:
            NumPy array on the CPU.
        """
        if not CUPY_AVAILABLE:
            return data
            
        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            logger.info(f"Transferring data to CPU: {data.shape} {data.dtype}")
            return cp.asnumpy(data)
        return data
    
    @staticmethod
    def matmul(a, b):
        """
        Performs matrix multiplication, leveraging GPU if CuPy arrays are provided.
        
        Args:
            a: First array (NumPy or CuPy).
            b: Second array (NumPy or CuPy).
            
        Returns:
            Result of matrix multiplication.
        """
        if CUPY_AVAILABLE and isinstance(a, cp.ndarray) and isinstance(b, cp.ndarray):
            logger.debug("Performing GPU matrix multiplication.")
            return cp.matmul(a, b)
        else:
            logger.debug("Performing CPU matrix multiplication.")
            return np.matmul(a, b)
            
    @staticmethod
    def norm(vec, axis: Optional[int] = None):
        """
        Computes the L2 norm of a vector or array, leveraging GPU if CuPy array is provided.
        
        Args:
            vec: Input vector or array (NumPy or CuPy).
            axis: Axis along which to compute the norm.
            
        Returns:
            L2 norm.
        """
        if CUPY_AVAILABLE and isinstance(vec, cp.ndarray):
            logger.debug("Performing GPU norm computation.")
            return cp.linalg.norm(vec, axis=axis)
        else:
            logger.debug("Performing CPU norm computation.")
            return np.linalg.norm(vec, axis=axis)
            
    @staticmethod
    def dot(a, b):
        """
        Performs dot product, leveraging GPU if CuPy arrays are provided.
        
        Args:
            a: First array (NumPy or CuPy).
            b: Second array (NumPy or CuPy).
            
        Returns:
            Result of dot product.
        """
        if CUPY_AVAILABLE and isinstance(a, cp.ndarray) and isinstance(b, cp.ndarray):
            logger.debug("Performing GPU dot product.")
            return cp.dot(a, b)
        else:
            logger.debug("Performing CPU dot product.")
            return np.dot(a, b)
            
    @staticmethod
    def is_gpu_available() -> bool:
        """
        Checks if a CUDA-enabled GPU is available.
        
        Returns:
            True if GPU is available, False otherwise.
        """
        if not CUPY_AVAILABLE:
            return False
        try:
            return cp.cuda.is_available()
        except:
            return False

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """
        Retrieves information about the available GPU.
        
        Returns:
            Dictionary containing GPU information.
        """
        if not GPUAccelerator.is_gpu_available():
            return {"status": "No CUDA-enabled GPU available"}
        
        device = cp.cuda.Device()
        return {
            "status": "CUDA-enabled GPU available",
            "name": device.name,
            "total_memory": device.mem_info[1],
            "free_memory": device.mem_info[0],
            "compute_capability": device.compute_capability
        }

    @staticmethod
    def synchronize():
        """
        Synchronizes the current CUDA device.
        """
        if GPUAccelerator.is_gpu_available():
            cp.cuda.Stream.null.synchronize()
            logger.debug("CUDA device synchronized.")


# Example usage (for testing purposes)
if __name__ == "__main__":
    if GPUAccelerator.is_gpu_available():
        print("GPU is available!")
        print(GPUAccelerator.get_gpu_info())
        
        # Create a NumPy array
        np_array = np.random.rand(1000, 1000).astype(np.float32)
        
        # Transfer to GPU
        cp_array = GPUAccelerator.to_gpu(np_array)
        
        # Perform matrix multiplication on GPU
        cp_result = GPUAccelerator.matmul(cp_array, cp_array)
        
        # Transfer back to CPU
        np_result = GPUAccelerator.to_cpu(cp_result)
        
        # Verify results
        assert np.allclose(np.matmul(np_array, np_array), np_result)
        print("GPU matrix multiplication verified.")
        
        # Test norm
        np_norm = GPUAccelerator.norm(np_array)
        cp_norm = GPUAccelerator.norm(cp_array)
        print(f"CPU norm: {np_norm:.4f}")
        print(f"GPU norm: {cp_norm:.4f}")
        assert np.allclose(np_norm, GPUAccelerator.to_cpu(cp_norm))
        print("GPU norm computation verified.")
        
    else:
        print("No CUDA-enabled GPU found. Running on CPU only.")
        np_array = np.random.rand(1000, 1000).astype(np.float32)
        np_result = GPUAccelerator.matmul(np_array, np_array)
        print("CPU matrix multiplication performed.")

