
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Union, Tuple
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class ApproxNNSearch:
    """
    Provides approximate nearest neighbor search capabilities using NearestNeighbors with KDTree.
    LSHForest was deprecated in scikit-learn 0.24 and removed in 1.0.
    """

    def __init__(self, n_neighbors_tree: int = 20, algorithm: str = 'kd_tree', random_state: Optional[int] = None):
        """
        Initializes the Approximate Nearest Neighbor Search with NearestNeighbors (KDTree or BallTree).

        Args:
            n_neighbors_tree: Number of neighbors to consider for tree construction.
            algorithm: Algorithm to use for nearest neighbor search (
                       'auto', 'ball_tree', 'kd_tree', 'brute').
            random_state: Seed for reproducibility (not directly used by NearestNeighbors, but good practice).
        """
        self.n_neighbors_tree = n_neighbors_tree
        self.algorithm = algorithm
        self.random_state = random_state
        self.nn_model = None # Will be initialized in fit
        logger.info(f"ApproxNNSearch initialized with algorithm={algorithm}")

    def fit(self, data_points: np.ndarray) -> None:
        """
        Fits the NearestNeighbors model to the given data points.

        Args:
            data_points: A 2D numpy array where each row is a data point.
        """
        try:
            # Use NearestNeighbors with a tree-based algorithm for approximate behavior
            self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors_tree, algorithm=self.algorithm)
            self.nn_model.fit(data_points)
            logger.info(f"NearestNeighbors model fitted to {data_points.shape[0]} data points using {self.algorithm}.")
        except Exception as e:
            logger.error(f"Error fitting NearestNeighbors model: {e}")
            raise

    def kneighbors(self, query_points: np.ndarray, n_neighbors: int = 5, return_distance: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Finds the k-nearest neighbors for the given query points.

        Args:
            query_points: A 2D numpy array where each row is a query point.
            n_neighbors: The number of nearest neighbors to return.
            return_distance: Whether to return distances along with indices.

        Returns:
            Distances and indices of the k-nearest neighbors, or just indices if return_distance is False.
        """
        if self.nn_model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        try:
            distances, indices = self.nn_model.kneighbors(query_points, n_neighbors=n_neighbors, return_distance=return_distance)
            logger.debug(f"Found {n_neighbors} nearest neighbors for {query_points.shape[0]} query points.")
            if return_distance:
                return distances, indices
            else:
                return indices
        except Exception as e:
            logger.error(f"Error finding k-neighbors: {e}")
            raise

    def set_params(self, **params) -> None:
        """
        Set the parameters of the NearestNeighbors estimator.

        Args:
            params: Estimator parameters.
        """
        if self.nn_model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        try:
            self.nn_model.set_params(**params)
            logger.info(f"NearestNeighbors parameters updated: {params}")
        except Exception as e:
            logger.error(f"Error setting NearestNeighbors parameters: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Generate some random data points
    np.random.seed(42)
    data_points = np.random.rand(1000, 10) # 1000 points in 10 dimensions
    query_point = np.random.rand(1, 10) # A single query point

    print("--- Testing ApproxNNSearch ---")
    
    # Initialize and fit the NN model with KDTree
    approx_nn_search = ApproxNNSearch(n_neighbors_tree=20, algorithm='kd_tree')
    approx_nn_search.fit(data_points)

    # Find k-nearest neighbors
    distances, indices = approx_nn_search.kneighbors(query_point, n_neighbors=5)
    print(f"Approximate distances to 5 nearest neighbors: {distances[0]}")
    print(f"Approximate indices of 5 nearest neighbors: {indices[0]}")
    
    # Verify with exact search for one point (for comparison)
    exact_nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(data_points)
    exact_distances, exact_indices = exact_nbrs.kneighbors(query_point)
    
    print(f"\nExact distances to 5 nearest neighbors: {exact_distances[0]}")
    print(f"Exact indices of 5 nearest neighbors: {exact_indices[0]}")
    
    # Check if approximate results are reasonably close to exact results
    # This is a loose check as tree-based methods are not always exact for ANN
    # but should be close for small k and well-behaved data.
    assert np.allclose(sorted(distances[0]), sorted(exact_distances[0]), atol=1e-5)
    # We can't assert exact index match due to approximation, but distances should be very close.
    print("Approximate NN search test successful!")

    print("All ApproxNNSearch tests passed!")


