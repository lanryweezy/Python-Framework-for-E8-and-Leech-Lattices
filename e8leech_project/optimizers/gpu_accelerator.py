import numpy as np
from typing import Union, Tuple, Dict, Any, Optional

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class GPUAccelerator:
    """
    Provides utilities for GPU acceleration using CuPy.
    """

    @staticmethod
    def to_gpu(data: np.ndarray):
        if not CUPY_AVAILABLE:
            return data

        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        return cp.asarray(data)

    @staticmethod
    def to_cpu(data):
        if not CUPY_AVAILABLE:
            return data

        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return data

    @staticmethod
    def matmul(a, b):
        if CUPY_AVAILABLE and isinstance(a, cp.ndarray) and isinstance(b, cp.ndarray):
            return cp.matmul(a, b)
        else:
            return np.matmul(a, b)

    @staticmethod
    def norm(vec, axis: Optional[int] = None):
        if CUPY_AVAILABLE and isinstance(vec, cp.ndarray):
            return cp.linalg.norm(vec, axis=axis)
        else:
            return np.linalg.norm(vec, axis=axis)

    @staticmethod
    def dot(a, b):
        if CUPY_AVAILABLE and isinstance(a, cp.ndarray) and isinstance(b, cp.ndarray):
            return cp.dot(a, b)
        else:
            return np.dot(a, b)

    @staticmethod
    def is_gpu_available() -> bool:
        if not CUPY_AVAILABLE:
            return False
        try:
            return cp.cuda.is_available()
        except:
            return False

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
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
        if GPUAccelerator.is_gpu_available():
            cp.cuda.Stream.null.synchronize()
