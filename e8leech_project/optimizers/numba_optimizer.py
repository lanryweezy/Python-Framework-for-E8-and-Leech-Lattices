import numpy as np
from numba import jit, prange
from typing import Callable, Optional

class NumbaOptimizer:
    """
    Provides utilities for Numba JIT compilation to optimize CPU-bound functions.
    """

    @staticmethod
    def jit_compile(func: Callable, signature: Optional[str] = None) -> Callable:
        if signature:
            compiled_func = jit(signature)(func)
        else:
            compiled_func = jit(nopython=True, parallel=False)(func)
        return compiled_func

    @staticmethod
    def parallel_jit_compile(func: Callable, signature: Optional[str] = None) -> Callable:
        if signature:
            compiled_func = jit(signature, nopython=True, parallel=True)(func)
        else:
            compiled_func = jit(nopython=True, parallel=True)(func)
        return compiled_func

    @staticmethod
    def vectorize(func: Callable, signature: Optional[str] = None) -> Callable:
        from numba import vectorize
        if signature:
            vectorized_func = vectorize(signature)(func)
        else:
            vectorized_func = vectorize(nopython=True)(func)
        return vectorized_func
