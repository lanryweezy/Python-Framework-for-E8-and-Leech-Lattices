"""
Memory Management module for efficient data representation.
"""

import numpy as np
from typing import Union

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Provides utilities for memory-efficient data representations, including FP16 quantization.
    """
    
    @staticmethod
    def to_fp16(data: np.ndarray) -> np.ndarray:
        """
        Converts a NumPy array to float16 (half-precision float) format.
        
        Args:
            data: Input NumPy array (e.g., float32, float64).
            
        Returns:
            NumPy array in float16 format.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        
        original_dtype = data.dtype
        if original_dtype == np.float16:
            logger.info("Data is already in float16 format.")
            return data
            
        logger.info(f"Converting data from {original_dtype} to float16: {data.shape}")
        return data.astype(np.float16)

    @staticmethod
    def from_fp16(data: np.ndarray, original_dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Converts a float16 NumPy array back to a higher precision float format.
        
        Args:
            data: Input NumPy array in float16 format.
            original_dtype: The desired original data type (e.g., np.float32, np.float64).
                            Defaults to np.float32.
                            
        Returns:
            NumPy array in the specified original_dtype format.
        """
        if not isinstance(data, np.ndarray) or data.dtype != np.float16:
            raise TypeError("Input must be a NumPy array of float16 dtype.")
            
        logger.info(f"Converting data from float16 to {original_dtype}: {data.shape}")
        return data.astype(original_dtype)

    @staticmethod
    def get_memory_usage(data: Union[np.ndarray, list, dict]) -> str:
        """
        Estimates the memory usage of a NumPy array, list, or dictionary.
        
        Args:
            data: The data structure to estimate memory usage for.
            
        Returns:
            A string representing the estimated memory usage (e.g., "1.2 MB").
        """
        if isinstance(data, np.ndarray):
            bytes_usage = data.nbytes
        elif isinstance(data, (list, dict)):
            # This is a rough estimate for lists/dicts, actual usage can vary
            import sys
            bytes_usage = sys.getsizeof(data)
            if isinstance(data, list):
                bytes_usage += sum(sys.getsizeof(item) for item in data)
            elif isinstance(data, dict):
                bytes_usage += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in data.items())
        else:
            raise TypeError("Unsupported data type for memory usage estimation.")
            
        # Convert bytes to human-readable format
        for unit in ["bytes", "KB", "MB", "GB", "TB"]:
            if bytes_usage < 1024.0:
                return f"{bytes_usage:.2f} {unit}"
            bytes_usage /= 1024.0
        return f"{bytes_usage:.2f} TB" # Should not reach here for typical data


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Create a large float64 array
    large_array_fp64 = np.random.rand(1000, 1000).astype(np.float64)
    print(f"Original (float64) memory usage: {MemoryManager.get_memory_usage(large_array_fp64)}")
    
    # Convert to float16
    large_array_fp16 = MemoryManager.to_fp16(large_array_fp64)
    print(f"FP16 memory usage: {MemoryManager.get_memory_usage(large_array_fp16)}")
    
    # Convert back to float32
    converted_back_fp32 = MemoryManager.from_fp16(large_array_fp16, np.float32)
    print(f"Converted back (float32) memory usage: {MemoryManager.get_memory_usage(converted_back_fp32)}")
    
    # Verify data integrity (with some tolerance due to precision loss)
    assert np.allclose(large_array_fp64, converted_back_fp32, atol=1e-3)
    print("Data conversion and integrity verified.")
    
    # Test with a list
    test_list = [np.random.rand(10,10) for _ in range(10)]
    print(f"List memory usage: {MemoryManager.get_memory_usage(test_list)}")
    
    # Test with a dictionary
    test_dict = {f"key_{i}": np.random.rand(5,5) for i in range(5)}
    print(f"Dictionary memory usage: {MemoryManager.get_memory_usage(test_dict)}")

