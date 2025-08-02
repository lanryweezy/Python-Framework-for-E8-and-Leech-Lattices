"""
Parallel Processing module using Ray
"""

import ray
import numpy as np
from typing import List, Callable, Any

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ParallelProcessor:
    """
    Provides utilities for parallel processing of lattice points using Ray.
    """
    
    def __init__(self):
        """
        Initializes the Ray cluster if not already initialized.
        """
        if not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True)
                logger.info("Ray initialized for parallel processing.")
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
                raise

    def process_in_parallel(self, func: Callable, data: List[Any]) -> List[Any]:
        """
        Distributes the processing of data using Ray.
        
        Args:
            func: The function to apply to each data item. This function should be
                  a standalone function or a static method of a class.
            data: A list of data items to be processed. Each item will be passed
                  as an argument to the function.
                  
        Returns:
            A list of results from processing each data item.
        """
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Call ParallelProcessor() first.")
            
        logger.info(f"Processing {len(data)} items in parallel using Ray.")
        
        # Create Ray remote function
        remote_func = ray.remote(func)
        
        # Submit tasks to Ray
        futures = [remote_func.remote(item) for item in data]
        
        # Get results
        results = ray.get(futures)
        logger.info("Parallel processing completed.")
        
        return results

    def shutdown(self):
        """
        Shuts down the Ray cluster.
        """
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray cluster shut down.")


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Define a simple function to be executed in parallel
    def square_and_add_one(x):
        return x**2 + 1

    processor = ParallelProcessor()
    
    test_data = list(range(100))
    
    results = processor.process_in_parallel(square_and_add_one, test_data)
    
    print("Original data:", test_data[:10])
    print("Processed results:", results[:10])
    
    expected_results = [x**2 + 1 for x in test_data]
    assert results == expected_results
    print("Parallel processing test passed.")
    
    processor.shutdown()

