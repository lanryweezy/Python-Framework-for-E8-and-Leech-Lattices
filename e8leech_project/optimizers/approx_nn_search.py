import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Union, Tuple

class ApproxNNSearch:
    """
    Provides approximate nearest neighbor search capabilities using NearestNeighbors with KDTree.
    """

    def __init__(self, n_neighbors_tree: int = 20, algorithm: str = 'kd_tree', random_state: Optional[int] = None):
        self.n_neighbors_tree = n_neighbors_tree
        self.algorithm = algorithm
        self.random_state = random_state
        self.nn_model = None

    def fit(self, data_points: np.ndarray) -> None:
        try:
            self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors_tree, algorithm=self.algorithm)
            self.nn_model.fit(data_points)
        except Exception as e:
            raise

    def kneighbors(self, query_points: np.ndarray, n_neighbors: int = 5, return_distance: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if self.nn_model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        try:
            distances, indices = self.nn_model.kneighbors(query_points, n_neighbors=n_neighbors, return_distance=return_distance)
            if return_distance:
                return distances, indices
            else:
                return indices
        except Exception as e:
            raise

    def set_params(self, **params) -> None:
        if self.nn_model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        try:
            self.nn_model.set_params(**params)
        except Exception as e:
            raise
