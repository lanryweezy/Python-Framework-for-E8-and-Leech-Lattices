import ray
import numpy as np
from typing import List, Callable, Any

class ParallelProcessor:
    """
    Provides utilities for parallel processing of lattice points using Ray.
    """

    def __init__(self):
        if not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True)
            except Exception as e:
                raise

    def process_in_parallel(self, func: Callable, data: List[Any]) -> List[Any]:
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Call ParallelProcessor() first.")

        remote_func = ray.remote(func)

        futures = [remote_func.remote(item) for item in data]

        results = ray.get(futures)

        return results

    def shutdown(self):
        if ray.is_initialized():
            ray.shutdown()
