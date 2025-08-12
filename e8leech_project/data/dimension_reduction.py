import numpy as np
from typing import Tuple, List, Optional, Dict

class DimensionReduction:
    """
    Lattice-based dimension reduction techniques.
    """

    def __init__(self, lattice_type: str = "E8", target_dimension: int = 3):
        self.lattice_type = lattice_type
        self.target_dimension = target_dimension
        self.lattice_basis = self._get_lattice_basis()
        self.source_dimension = len(self.lattice_basis)
        self.projection_matrix = None
        self.reconstruction_matrix = None
        self.mean_vector = None

    def _get_lattice_basis(self) -> np.ndarray:
        if self.lattice_type == "E8":
            basis = np.zeros((8, 8))
            for i in range(7):
                basis[i, i] = 1.0
                basis[i, i+1] = -1.0
            basis[7, :] = 0.5
            return basis
        elif self.lattice_type == "Leech":
            return np.eye(24)
        else:
            return np.eye(8)

    def fit(self, data: np.ndarray) -> 'DimensionReduction':
        self.mean_vector = np.mean(data, axis=0)
        centered_data = data - self.mean_vector
        cov_matrix = np.cov(centered_data.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        self.projection_matrix = sorted_eigenvectors[:, :self.target_dimension].T
        self.reconstruction_matrix = self.projection_matrix.T
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.projection_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        centered_data = data - self.mean_vector
        return centered_data @ self.projection_matrix.T

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        if self.reconstruction_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return transformed_data @ self.reconstruction_matrix.T + self.mean_vector

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)
