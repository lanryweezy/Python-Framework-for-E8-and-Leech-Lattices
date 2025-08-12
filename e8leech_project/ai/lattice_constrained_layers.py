import numpy as np
from typing import Optional, Callable

class LatticeConstrainedLayer:
    """
    Neural network layer with lattice constraints.
    """

    def __init__(self, input_dim: int, output_dim: int, lattice_type: str = "E8",
                 constraint_strength: float = 1.0, activation: str = "relu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lattice_type = lattice_type
        self.constraint_strength = constraint_strength
        self.activation = activation

        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.bias = np.zeros(output_dim)

        self.lattice_basis = self._get_lattice_basis()
        self.lattice_points = self._generate_lattice_points()

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
        elif self.lattice_type == "Z":
            dim = min(self.input_dim, self.output_dim)
            return np.eye(dim)
        else:
            dim = min(self.input_dim, self.output_dim)
            return np.eye(dim)

    def _generate_lattice_points(self, num_points: int = 100) -> np.ndarray:
        if self.lattice_type == "E8":
            points = []
            for i in range(8):
                for j in range(i + 1, 8):
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            point = np.zeros(8)
                            point[i] = s1
                            point[j] = s2
                            points.append(point)
                            if len(points) >= num_points:
                                return np.array(points)
            return np.array(points)
        elif self.lattice_type == "Leech":
            return np.random.randint(-2, 3, size=(num_points, 24))
        elif self.lattice_type == "Z":
            dim = min(self.input_dim, self.output_dim)
            return np.random.randint(-3, 4, size=(num_points, dim))
        else:
            dim = min(self.input_dim, self.output_dim)
            return np.random.randn(num_points, dim)

    def _project_to_lattice(self, x: np.ndarray) -> np.ndarray:
        if self.lattice_type == "Z":
            return np.round(x)
        elif self.lattice_type in ["E8", "Leech"]:
            if len(self.lattice_points) == 0:
                return x
            distances = np.linalg.norm(self.lattice_points - x, axis=1)
            nearest_idx = np.argmin(distances)
            return self.lattice_points[nearest_idx]
        else:
            return x

    def _apply_lattice_constraint(self, weights: np.ndarray) -> np.ndarray:
        if self.constraint_strength == 0:
            return weights
        constrained_weights = weights.copy()
        for i in range(weights.shape[1]):
            weight_vector = weights[:, i]
            if len(weight_vector) <= len(self.lattice_basis):
                projected = self._project_to_lattice(weight_vector[:len(self.lattice_basis)])
                constrained_vector = (1 - self.constraint_strength) * weight_vector
                constrained_vector[:len(projected)] += self.constraint_strength * projected
                constrained_weights[:, i] = constrained_vector
        return constrained_weights

    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        constrained_weights = self._apply_lattice_constraint(self.weights)
        output = x @ constrained_weights + self.bias
        output = self._activation_function(output)
        if self.constraint_strength > 0.5:
            for i in range(output.shape[0]):
                if output.shape[1] <= len(self.lattice_basis):
                    projected = self._project_to_lattice(output[i, :len(self.lattice_basis)])
                    output[i, :len(projected)] = projected
        return output
