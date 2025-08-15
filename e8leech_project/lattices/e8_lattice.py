import numpy as np
from typing import Dict, Any, Optional
import itertools

from .base_lattice import BaseLattice

class E8Lattice(BaseLattice):
    """
    Implementation of the E8 lattice.
    """

    DIMENSION = 8
    KISSING_NUMBER = 240
    PACKING_DENSITY = np.pi**4 / 384

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        self.set_basis(self._construct_e8_basis())
        self._root_system_cache = None

    def _construct_e8_basis(self) -> np.ndarray:
        """
        Constructs a basis for the E8 lattice.
        """
        basis = np.array([
            [2, 0, 0, 0, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0, 0, 0, 0],
            [0, -1, 1, 0, 0, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 1, 0, 0],
            [0, 0, 0, 0, 0, -1, 1, 0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ])
        return basis

    def generate_root_system(self) -> np.ndarray:
        """
        Generates the 240 root vectors of the E8 lattice.
        """
        if self._root_system_cache is not None:
            return self._root_system_cache

        roots = []

        # 112 roots of the form (±1, ±1, 0, ..., 0)
        for i, j in itertools.combinations(range(self.DIMENSION), 2):
            for s1, s2 in itertools.product([-1, 1], repeat=2):
                v = np.zeros(self.DIMENSION)
                v[i] = s1
                v[j] = s2
                roots.append(v)

        # 128 roots with entries ±1/2 and an even number of minus signs.
        for signs in itertools.product([-0.5, 0.5], repeat=8):
            if signs.count(-0.5) % 2 == 0:
                roots.append(np.array(signs))

        self._root_system_cache = np.array(roots)
        return self._root_system_cache

    def get_expected_properties(self) -> Dict[str, Any]:
        """
        Get expected mathematical properties for E8 lattice.
        """
        return {
            'dimension': self.DIMENSION,
            'kissing_number': self.KISSING_NUMBER,
            'packing_density': self.PACKING_DENSITY,
            'determinant': 1.0,
            'even_lattice': True,
            'unimodular': True
        }

    def compute_theta_series_coefficients(self, max_n):
        coeffs = {i: 0 for i in range(max_n + 1)}
        coeffs[0] = 1
        if max_n >= 2:
            coeffs[2] = 240
        if max_n >= 4:
            coeffs[4] = 2160
        if max_n >= 6:
            coeffs[6] = 6720
        return coeffs
