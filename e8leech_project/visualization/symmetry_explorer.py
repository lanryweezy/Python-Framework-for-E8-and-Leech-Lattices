import numpy as np
from typing import Tuple, List, Optional, Dict, Set
import json
from itertools import permutations, combinations

class SymmetryExplorer:
    """
    Interactive explorer for lattice symmetry groups.
    """

    def __init__(self, lattice_type: str = "E8"):
        self.lattice_type = lattice_type
        self.lattice_basis = self._get_lattice_basis()
        self.dimension = len(self.lattice_basis)
        self.generators = self._get_symmetry_generators()

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
            return np.eye(3)

    def _get_symmetry_generators(self) -> List[np.ndarray]:
        # Simplified generator fetching
        return [np.eye(self.dimension)]

    def compute_orbit(self, point: np.ndarray) -> List[np.ndarray]:
        orbit = [point]
        for g in self.generators:
            orbit.append(point @ g.T)
        return orbit

    def compute_stabilizer(self, point: np.ndarray) -> List[np.ndarray]:
        stabilizer = []
        for g in self.generators:
            if np.allclose(point @ g.T, point):
                stabilizer.append(g)
        return stabilizer

    def export_symmetry_data(self, point: np.ndarray, filename: str) -> Dict:
        orbit = self.compute_orbit(point)
        stabilizer = self.compute_stabilizer(point)
        symmetry_data = {
            "point": point.tolist(),
            "orbit": [p.tolist() for p in orbit],
            "stabilizer": [s.tolist() for s in stabilizer]
        }
        with open(filename, 'w') as f:
            json.dump(symmetry_data, f, indent=2)
        return symmetry_data
