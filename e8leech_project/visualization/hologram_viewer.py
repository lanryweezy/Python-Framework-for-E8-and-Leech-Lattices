import numpy as np
from typing import Tuple, List, Optional, Dict, Callable
import json

class HologramViewer:
    """
    Interactive holographic viewer for high-dimensional lattice structures.
    """

    def __init__(self, lattice_type: str = "Leech", projection_dim: int = 3):
        self.lattice_type = lattice_type
        self.projection_dim = projection_dim
        self.lattice_basis = self._get_lattice_basis()
        self.source_dimension = len(self.lattice_basis)
        self.projection_matrices = self._initialize_projection_matrices()

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

    def _initialize_projection_matrices(self) -> List[np.ndarray]:
        projections = []
        for _ in range(10):
            projection = np.random.randn(self.projection_dim, self.source_dimension)
            projections.append(projection)
        return projections

    def project_points(self, points: np.ndarray, projection_index: int = 0) -> np.ndarray:
        if projection_index >= len(self.projection_matrices):
            projection_index = 0
        projection_matrix = self.projection_matrices[projection_index]
        return points @ projection_matrix.T

    def create_slice(self, points: np.ndarray, slice_dimension: int, slice_value: float, thickness: float = 1.0) -> np.ndarray:
        if slice_dimension >= self.source_dimension:
            slice_dimension = 0
        slice_mask = np.abs(points[:, slice_dimension] - slice_value) <= thickness / 2
        return points[slice_mask]

    def export_visualization_data(self, points: np.ndarray, filename: str, projection_index: int = 0) -> Dict:
        projected_points = self.project_points(points, projection_index)
        viz_data = {
            "projected_points": projected_points.tolist(),
        }
        with open(filename, 'w') as f:
            json.dump(viz_data, f, indent=2)
        return viz_data
