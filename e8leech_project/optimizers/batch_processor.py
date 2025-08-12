import numpy as np
from typing import List, Callable, Any, Generator

class BatchProcessor:
    """
    Provides utilities for processing data in batches.
    """

    @staticmethod
    def create_batches(data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
        if not isinstance(data, list):
            raise TypeError("Input data must be a list.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")

        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    @staticmethod
    def process_batches(data: List[Any], batch_size: int, process_func: Callable) -> List[Any]:
        if not callable(process_func):
            raise TypeError("process_func must be a callable function.")

        all_results = []
        for i, batch in enumerate(BatchProcessor.create_batches(data, batch_size)):
            batch_results = process_func(batch)
            all_results.extend(batch_results)

        return all_results
