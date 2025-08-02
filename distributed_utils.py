"""
Distributed computing utilities for E8 Leech Lattice Framework.

This module provides utilities for leveraging Dask and Ray for distributed
computation, enabling scalable processing of large lattice datasets and
complex simulations.
"""

import os
import numpy as np
from typing import Any, Dict, List, Callable, Union, Tuple # Added Tuple
from .logging_utils import get_logger

logger = get_logger(__name__)

# Check if Dask is available and import da if so
DASK_AVAILABLE = False
try:
    import dask.array as da
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    logger.warning("Dask not available. Install with: pip install dask distributed")

# Check if Ray is available and import ray if so
RAY_AVAILABLE = False
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    logger.warning("Ray not available. Install with: pip install ray")


class DistributedProcessor:
    """
    Manages distributed computation using Dask or Ray.
    """
    
    def __init__(self, backend: str = "ray", num_workers: int = None, **kwargs):
        """
        Initialize the distributed processor.
        
        Args:
            backend: The distributed computing backend to use ("dask" or "ray").
            num_workers: Number of workers to use. If None, uses all available cores.
            kwargs: Additional arguments for backend initialization.
        """
        self.backend = backend
        self.client = None
        self.cluster = None
        
        if self.backend == "dask":
            if not DASK_AVAILABLE:
                raise ImportError("Dask is not available. Please install it.")
            self._init_dask(num_workers, **kwargs)
        elif self.backend == "ray":
            if not RAY_AVAILABLE:
                raise ImportError("Ray is not available. Please install it.")
            self._init_ray(num_workers, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Choose \'dask\' or \'ray\'.")
        
        logger.info(f"Initialized DistributedProcessor with backend: {self.backend}")

    def _init_dask(self, num_workers: int, **kwargs):
        """
        Initialize Dask client and cluster.
        """
        try:
            self.cluster = LocalCluster(n_workers=num_workers, **kwargs)
            self.client = Client(self.cluster)
            logger.info(f"Dask client initialized: {self.client.dashboard_link}")
        except Exception as e:
            logger.error(f"Error initializing Dask: {e}")
            raise

    def _init_ray(self, num_workers: int, **kwargs):
        """
        Initialize Ray.
        """
        try:
            if not ray.is_initialized():
                ray.init(num_cpus=num_workers, **kwargs)
            self.client = ray
            logger.info(f"Ray initialized. Dashboard: {ray.get_webui_url()}")
        except Exception as e:
            logger.error(f"Error initializing Ray: {e}")
            raise

    def map(self, func: Callable, data: List[Any]) -> List[Any]:
        """
        Apply a function to each item in a list in parallel.
        
        Args:
            func: The function to apply.
            data: A list of data items.
            
        Returns:
            A list of results.
        """
        if self.backend == "dask":
            futures = self.client.map(func, data)
            results = self.client.gather(futures)
        elif self.backend == "ray":
            @ray.remote
            def remote_func(item):
                return func(item)
            futures = [remote_func.remote(item) for item in data]
            results = ray.get(futures)
        else:
            results = [func(item) for item in data] # Fallback to serial
        
        logger.debug(f"Mapped function across {len(data)} items using {self.backend}")
        return results

    def compute(self, dask_array: Any) -> np.ndarray:
        """
        Compute a Dask array, returning a NumPy array.
        
        Args:
            dask_array: The Dask array to compute.
            
        Returns:
            A NumPy array with the computed results.
        """
        if self.backend == "dask":
            if not DASK_AVAILABLE:
                raise ImportError("Dask is not available.")
            result = dask_array.compute()
            logger.debug("Dask array computed.")
            return result
        else:
            raise ValueError("Compute method is only applicable for Dask backend.")

    def from_numpy(self, array: np.ndarray, chunks: Union[int, Tuple[int, ...], str] = "auto") -> Any:
        """
        Convert a NumPy array to a Dask array.
        
        Args:
            array: The NumPy array to convert.
            chunks: Chunking scheme for the Dask array.
            
        Returns:
            A Dask array.
        """
        if self.backend == "dask":
            if not DASK_AVAILABLE:
                raise ImportError("Dask is not available.")
            return da.from_array(array, chunks=chunks)
        else:
            raise ValueError("from_numpy method is only applicable for Dask backend.")

    def shutdown(self):
        """
        Shut down the distributed computing client and cluster.
        """
        if self.client:
            if self.backend == "dask":
                self.client.close()
                if self.cluster:
                    self.cluster.close()
                logger.info("Dask client and cluster shut down.")
            elif self.backend == "ray":
                ray.shutdown()
                logger.info("Ray shut down.")
            self.client = None
            self.cluster = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# Example usage and testing
if __name__ == "__main__":
    def process_item(x):
        # Simulate some computation
        return x * x + 1

    data_to_process = list(range(100))

    print("\n--- Testing Ray Backend ---")
    if RAY_AVAILABLE:
        try:
            with DistributedProcessor(backend="ray", num_workers=4) as processor:
                results_ray = processor.map(process_item, data_to_process)
                print(f"Ray results (first 5): {results_ray[:5]}")
                assert len(results_ray) == len(data_to_process)
                assert results_ray[0] == process_item(0)
                print("Ray backend test successful!")
        except Exception as e:
            print(f"Ray backend test failed: {e}")
    else:
        print("Ray not available, skipping Ray tests.")

    print("\n--- Testing Dask Backend ---")
    if DASK_AVAILABLE:
        try:
            with DistributedProcessor(backend="dask", num_workers=4) as processor:
                results_dask = processor.map(process_item, data_to_process)
                print(f"Dask results (first 5): {results_dask[:5]}")
                assert len(results_dask) == len(data_to_process)
                assert results_dask[0] == process_item(0)
                print("Dask map test successful!")

                # Test Dask array computation
                np_array = np.arange(100).reshape(10, 10)
                dask_arr = processor.from_numpy(np_array, chunks=(5, 5))
                computed_arr = processor.compute(dask_arr * 2)
                print(f"Dask array computed (first 5): {computed_arr.flatten()[:5]}")
                assert np.array_equal(computed_arr, np_array * 2)
                print("Dask array computation test successful!")

        except Exception as e:
            print(f"Dask backend test failed: {e}")
    else:
        print("Dask not available, skipping Dask tests.")

    print("All distributed utilities tests completed!")

