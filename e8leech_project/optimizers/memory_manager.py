import numpy as np
from typing import Union

class MemoryManager:
    """
    Provides utilities for memory-efficient data representations, including FP16 quantization.
    """

    @staticmethod
    def to_fp16(data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        return data.astype(np.float16)

    @staticmethod
    def from_fp16(data: np.ndarray, original_dtype: np.dtype = np.float32) -> np.ndarray:
        if not isinstance(data, np.ndarray) or data.dtype != np.float16:
            raise TypeError("Input must be a NumPy array of float16 dtype.")

        return data.astype(original_dtype)

    @staticmethod
    def get_memory_usage(data: Union[np.ndarray, list, dict]) -> str:
        if isinstance(data, np.ndarray):
            bytes_usage = data.nbytes
        elif isinstance(data, (list, dict)):
            import sys
            bytes_usage = sys.getsizeof(data)
            if isinstance(data, list):
                bytes_usage += sum(sys.getsizeof(item) for item in data)
            elif isinstance(data, dict):
                bytes_usage += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in data.items())
        else:
            raise TypeError("Unsupported data type for memory usage estimation.")

        for unit in ["bytes", "KB", "MB", "GB", "TB"]:
            if bytes_usage < 1024.0:
                return f"{bytes_usage:.2f} {unit}"
            bytes_usage /= 1024.0
        return f"{bytes_usage:.2f} TB"
