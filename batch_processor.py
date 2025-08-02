"""
Batch Processing module for AI/quantum applications.
"""

import numpy as np
from typing import List, Callable, Any, Generator

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class BatchProcessor:
    """
    Provides utilities for processing data in batches, which is crucial for
    optimizing performance in AI and quantum computing applications.
    """
    
    @staticmethod
    def create_batches(data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
        """
        Generates batches from a list of data.
        
        Args:
            data: The list of data items to be batched.
            batch_size: The maximum size of each batch.
            
        Yields:
            A list representing a batch of data.
        """
        if not isinstance(data, list):
            raise TypeError("Input data must be a list.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
            
        logger.info(f"Creating batches of size {batch_size} from {len(data)} items.")
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    @staticmethod
    def process_batches(data: List[Any], batch_size: int, process_func: Callable) -> List[Any]:
        """
        Processes data in batches using a specified function.
        
        Args:
            data: The list of data items to be processed.
            batch_size: The size of each batch.
            process_func: A callable function that takes a batch of data as input
                          and returns a list of results for that batch.
                          
        Returns:
            A flattened list of all results from processing the batches.
        """
        if not callable(process_func):
            raise TypeError("process_func must be a callable function.")
            
        all_results = []
        for i, batch in enumerate(BatchProcessor.create_batches(data, batch_size)):
            logger.debug(f"Processing batch {i+1} with {len(batch)} items.")
            batch_results = process_func(batch)
            all_results.extend(batch_results)
            
        logger.info(f"Processed {len(data)} items in batches. Total results: {len(all_results)}")
        return all_results


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Example processing function
    def simulate_quantum_operation(batch_of_vectors: List[np.ndarray]) -> List[np.ndarray]:
        # Simulate some quantum operation (e.g., applying a gate, measuring)
        # For demonstration, we'll just multiply by a scalar and add noise
        results = []
        for vec in batch_of_vectors:
            processed_vec = vec * 0.5 + np.random.rand(*vec.shape) * 0.1
            results.append(processed_vec)
        return results

    # Generate some dummy lattice points
    np.random.seed(42)
    lattice_points = [np.random.rand(24) for _ in range(1000)]
    
    batch_size = 128
    
    print(f"Total lattice points: {len(lattice_points)}")
    
    # Test create_batches
    print("\nTesting create_batches:")
    batches = list(BatchProcessor.create_batches(lattice_points, batch_size))
    print(f"Number of batches created: {len(batches)}")
    print(f"Size of first batch: {len(batches[0])}")
    print(f"Size of last batch: {len(batches[-1])}")
    
    # Test process_batches
    print("\nTesting process_batches:")
    processed_points = BatchProcessor.process_batches(lattice_points, batch_size, simulate_quantum_operation)
    
    print(f"Total processed points: {len(processed_points)}")
    print(f"Shape of a processed point: {processed_points[0].shape}")
    
    # Verify that the number of processed points matches original
    assert len(processed_points) == len(lattice_points)
    print("Batch processing test passed.")

