import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseLattice(ABC):
    """
    Abstract base class for lattice implementations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.basis = None
        self.gram_matrix = None
        self.determinant = None

    def set_basis(self, basis: np.ndarray):
        """
        Sets the basis for the lattice.
        """
        self.basis = basis
        self._gram_matrix = self._basis @ self._basis.T
        
        self._inverse_gram_matrix = np.linalg.inv(self._gram_matrix)

    @abstractmethod
    def generate_root_system(self) -> np.ndarray:
        """
        Generates the root system of the lattice.
        """
        pass

    @abstractmethod
    def get_expected_properties(self) -> Dict[str, Any]:
        """
        Returns a dictionary of expected mathematical properties of the lattice.
        """
        pass

    def quantize(self, vectors: np.ndarray, algorithm: str = 'babai') -> np.ndarray:
        """
        Quantizes vectors to the nearest lattice points.
        """
        if self.basis is None:
            raise ValueError("Basis not set.")

        if algorithm == 'babai':
            # For multiple vectors, we need to apply babai to each
            return np.array([self.babai_nearest_plane(v) for v in vectors])
        else:
            raise NotImplementedError(f"Algorithm '{algorithm}' not implemented.")


    def nearest_neighbor(self, target: np.ndarray, algorithm: str = 'babai') -> np.ndarray:
        """
        Finds the nearest lattice point to a target vector.
        """
        if algorithm == 'babai':
            return self.babai_nearest_plane(target)
        else:
            raise NotImplementedError(f"Algorithm '{algorithm}' not implemented.")

    def babai_nearest_plane(self, target: np.ndarray) -> np.ndarray:
        """
        Babai's nearest plane algorithm for finding the closest lattice point.
        """
        if self.basis is None:
            raise ValueError("Basis not set.")

        b = self.basis.T
        target_coords = np.linalg.solve(b, target)
        rounded_coords = np.round(target_coords)
        nearest_point = b @ rounded_coords
        return nearest_point

    def compute_theta_function(self, tau: complex, max_terms: int = 100) -> complex:
        """
        Computes the theta function of the lattice.
        """
        if self.basis is None:
            raise ValueError("Basis not set.")

        root_system = self.generate_root_system()

        theta_sum = 0.0
        for v in root_system:
            norm_sq = np.dot(v, v)
            theta_sum += np.exp(1j * np.pi * tau * norm_sq)

        return theta_sum
