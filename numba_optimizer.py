
import numpy as np
from numba import jit, prange
from typing import Callable, Optional
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class NumbaOptimizer:
    """
    Provides utilities for Numba JIT compilation to optimize CPU-bound functions.
    """

    @staticmethod
    def jit_compile(func: Callable, signature: Optional[str] = None) -> Callable:
        """
        Applies Numba JIT compilation to a given function.

        Args:
            func: The function to be JIT compiled.
            signature: Optional Numba signature string for ahead-of-time (AOT) compilation.
                       e.g., "float64(float64[:])"

        Returns:
            The JIT compiled function.
        """
        if signature:
            compiled_func = jit(signature)(func)
        else:
            compiled_func = jit(nopython=True, parallel=False)(func)
        logger.debug(f"Function {func.__name__} JIT compiled.")
        return compiled_func

    @staticmethod
    def parallel_jit_compile(func: Callable, signature: Optional[str] = None) -> Callable:
        """
        Applies Numba JIT compilation with parallel execution to a given function.

        Args:
            func: The function to be JIT compiled for parallel execution.
            signature: Optional Numba signature string.

        Returns:
            The JIT compiled function with parallel support.
        """
        if signature:
            compiled_func = jit(signature, nopython=True, parallel=True)(func)
        else:
            compiled_func = jit(nopython=True, parallel=True)(func)
        logger.debug(f"Function {func.__name__} JIT compiled with parallel support.")
        return compiled_func

    @staticmethod
    def vectorize(func: Callable, signature: Optional[str] = None) -> Callable:
        """
        Applies Numba vectorize to a given function for element-wise operations.

        Args:
            func: The function to be vectorized.
            signature: Optional Numba signature string.

        Returns:
            The vectorized function.
        """
        from numba import vectorize
        if signature:
            vectorized_func = vectorize(signature)(func)
        else:
            vectorized_func = vectorize(nopython=True)(func)
        logger.debug(f"Function {func.__name__} vectorized.")
        return vectorized_func


# Example usage and testing
if __name__ == "__main__":
    def sum_array(arr):
        total = 0.0
        for x in arr:
            total += x
        return total

    def multiply_arrays(arr1, arr2):
        result = np.empty_like(arr1)
        for i in prange(len(arr1)):
            result[i] = arr1[i] * arr2[i]
        return result

    def add_one(x):
        return x + 1

    # Test jit_compile
    compiled_sum_array = NumbaOptimizer.jit_compile(sum_array, "float64(float64[:])")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result_sum = compiled_sum_array(data)
    print(f"Compiled sum_array result: {result_sum}")
    assert np.isclose(result_sum, np.sum(data))

    # Test parallel_jit_compile
    compiled_multiply_arrays = NumbaOptimizer.parallel_jit_compile(multiply_arrays, "float64[:](float64[:], float64[:])")
    arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    arr2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    result_multiply = compiled_multiply_arrays(arr1, arr2)
    print(f"Compiled multiply_arrays result: {result_multiply}")
    assert np.allclose(result_multiply, arr1 * arr2)

    # Test vectorize
    vectorized_add_one = NumbaOptimizer.vectorize(add_one, "float64(float64)")
    data_vec = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    result_vectorize = vectorized_add_one(data_vec)
    print(f"Vectorized add_one result: {result_vectorize}")
    assert np.allclose(result_vectorize, data_vec + 1)

    print("All NumbaOptimizer tests passed!")


