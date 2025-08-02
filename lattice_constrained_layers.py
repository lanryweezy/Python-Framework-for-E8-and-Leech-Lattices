"""
Lattice-constrained neural network layers.

This module implements neural network layers that are constrained to respect
lattice structures, enabling more structured and interpretable representations.
"""

import numpy as np
from typing import Optional, Callable
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class LatticeConstrainedLayer:
    """
    Neural network layer with lattice constraints.
    
    This layer constrains weights and activations to respect lattice structures,
    which can improve generalization and interpretability.
    """
    
    def __init__(self, input_dim: int, output_dim: int, lattice_type: str = "E8",
                 constraint_strength: float = 1.0, activation: str = "relu"):
        """
        Initialize lattice-constrained layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            lattice_type: Type of lattice constraint ("E8", "Leech", or "Z")
            constraint_strength: Strength of lattice constraint (0 = no constraint, 1 = full constraint)
            activation: Activation function ("relu", "tanh", "sigmoid", "linear")
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lattice_type = lattice_type
        self.constraint_strength = constraint_strength
        self.activation = activation
        
        # Initialize weights
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.bias = np.zeros(output_dim)
        
        # Set up lattice structure
        self.lattice_basis = self._get_lattice_basis()
        self.lattice_points = self._generate_lattice_points()
        
        logger.info(f"Initialized lattice-constrained layer: {input_dim} -> {output_dim}")
        logger.info(f"Lattice type: {lattice_type}, constraint strength: {constraint_strength}")
    
    def _get_lattice_basis(self) -> np.ndarray:
        """
        Get basis vectors for the specified lattice.
        
        Returns:
            Lattice basis matrix
        """
        if self.lattice_type == "E8":
            # E8 lattice basis
            basis = np.zeros((8, 8))
            for i in range(7):
                basis[i, i] = 1.0
                basis[i, i+1] = -1.0
            basis[7, :] = 0.5
            return basis
        
        elif self.lattice_type == "Leech":
            # Simplified Leech lattice basis (identity for demonstration)
            return np.eye(24)
        
        elif self.lattice_type == "Z":
            # Integer lattice
            dim = min(self.input_dim, self.output_dim)
            return np.eye(dim)
        
        else:
            # Default to identity
            dim = min(self.input_dim, self.output_dim)
            return np.eye(dim)
    
    def _generate_lattice_points(self, num_points: int = 100) -> np.ndarray:
        """
        Generate lattice points for constraint enforcement.
        
        Args:
            num_points: Number of lattice points to generate
            
        Returns:
            Array of lattice points
        """
        if self.lattice_type == "E8":
            # Generate E8 lattice points
            points = []
            
            # Add some root vectors
            for i in range(8):
                for j in range(i + 1, 8):
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            point = np.zeros(8)
                            point[i] = s1
                            point[j] = s2
                            points.append(point)
                            if len(points) >= num_points:
                                break
                        if len(points) >= num_points:
                            break
                    if len(points) >= num_points:
                        break
                if len(points) >= num_points:
                    break
            
            return np.array(points[:num_points])
        
        elif self.lattice_type == "Leech":
            # Generate simplified Leech lattice points
            points = []
            for _ in range(num_points):
                point = np.random.randint(-2, 3, size=24)
                points.append(point)
            return np.array(points)
        
        elif self.lattice_type == "Z":
            # Integer lattice points
            dim = min(self.input_dim, self.output_dim)
            points = []
            for _ in range(num_points):
                point = np.random.randint(-3, 4, size=dim)
                points.append(point)
            return np.array(points)
        
        else:
            # Random points
            dim = min(self.input_dim, self.output_dim)
            return np.random.randn(num_points, dim)
    
    def _project_to_lattice(self, x: np.ndarray) -> np.ndarray:
        """
        Project vector to nearest lattice point.
        
        Args:
            x: Input vector
            
        Returns:
            Projected vector
        """
        if self.lattice_type == "Z":
            # Project to integer lattice
            return np.round(x)
        
        elif self.lattice_type in ["E8", "Leech"]:
            # Find nearest lattice point (simplified)
            if len(self.lattice_points) == 0:
                return x
            
            distances = np.linalg.norm(self.lattice_points - x, axis=1)
            nearest_idx = np.argmin(distances)
            return self.lattice_points[nearest_idx]
        
        else:
            return x
    
    def _apply_lattice_constraint(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply lattice constraints to weights.
        
        Args:
            weights: Input weights
            
        Returns:
            Constrained weights
        """
        if self.constraint_strength == 0:
            return weights
        
        constrained_weights = weights.copy()
        
        # Apply constraint to each weight vector
        for i in range(weights.shape[1]):
            weight_vector = weights[:, i]
            
            # Project to lattice if dimension matches
            if len(weight_vector) <= len(self.lattice_basis):
                projected = self._project_to_lattice(weight_vector[:len(self.lattice_basis)])
                
                # Interpolate between original and projected weights
                constrained_vector = (1 - self.constraint_strength) * weight_vector
                constrained_vector[:len(projected)] += self.constraint_strength * projected
                
                constrained_weights[:, i] = constrained_vector
        
        return constrained_weights
    
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """
        Apply activation function.
        
        Args:
            x: Input array
            
        Returns:
            Activated output
        """
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == "linear":
            return x
        else:
            return x
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the lattice-constrained layer.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Apply lattice constraints to weights
        constrained_weights = self._apply_lattice_constraint(self.weights)
        
        # Linear transformation
        output = x @ constrained_weights + self.bias
        
        # Apply activation
        output = self._activation_function(output)
        
        # Optionally apply lattice constraints to output
        if self.constraint_strength > 0.5:  # Only for strong constraints
            batch_size = output.shape[0]
            for i in range(batch_size):
                if output.shape[1] <= len(self.lattice_basis):
                    projected = self._project_to_lattice(output[i, :len(self.lattice_basis)])
                    output[i, :len(projected)] = projected
        
        logger.debug(f"Forward pass completed. Output shape: {output.shape}")
        return output
    
    def get_lattice_regularization_loss(self) -> float:
        """
        Compute regularization loss based on lattice constraints.
        
        Returns:
            Regularization loss
        """
        if self.constraint_strength == 0:
            return 0.0
        
        total_loss = 0.0
        
        # Compute distance from weights to nearest lattice points
        for i in range(self.weights.shape[1]):
            weight_vector = self.weights[:, i]
            
            if len(weight_vector) <= len(self.lattice_basis):
                projected = self._project_to_lattice(weight_vector[:len(self.lattice_basis)])
                distance = np.linalg.norm(weight_vector[:len(projected)] - projected)
                total_loss += distance ** 2
        
        return total_loss * self.constraint_strength
    
    def update_weights(self, grad_weights: np.ndarray, grad_bias: np.ndarray, 
                      learning_rate: float = 0.01):
        """
        Update weights with gradient descent and lattice constraints.
        
        Args:
            grad_weights: Weight gradients
            grad_bias: Bias gradients
            learning_rate: Learning rate
        """
        # Standard gradient update
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        # Apply lattice constraints
        self.weights = self._apply_lattice_constraint(self.weights)
        
        logger.debug("Weights updated with lattice constraints")
    
    def get_constraint_info(self) -> dict:
        """
        Get information about the lattice constraints.
        
        Returns:
            Dictionary with constraint information
        """
        return {
            "lattice_type": self.lattice_type,
            "constraint_strength": self.constraint_strength,
            "lattice_basis_shape": self.lattice_basis.shape,
            "num_lattice_points": len(self.lattice_points),
            "regularization_loss": self.get_lattice_regularization_loss()
        }


class LatticeConstrainedMLP:
    """
    Multi-layer perceptron with lattice constraints.
    """
    
    def __init__(self, layer_dims: list, lattice_type: str = "E8", 
                 constraint_strength: float = 1.0):
        """
        Initialize lattice-constrained MLP.
        
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden_dim1, ..., output_dim]
            lattice_type: Type of lattice constraint
            constraint_strength: Strength of lattice constraint
        """
        self.layer_dims = layer_dims
        self.lattice_type = lattice_type
        self.constraint_strength = constraint_strength
        
        # Create layers
        self.layers = []
        for i in range(len(layer_dims) - 1):
            activation = "relu" if i < len(layer_dims) - 2 else "linear"
            layer = LatticeConstrainedLayer(
                layer_dims[i], layer_dims[i + 1], 
                lattice_type, constraint_strength, activation
            )
            self.layers.append(layer)
        
        logger.info(f"Initialized lattice-constrained MLP with {len(self.layers)} layers")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def get_total_regularization_loss(self) -> float:
        """
        Get total regularization loss from all layers.
        
        Returns:
            Total regularization loss
        """
        total_loss = 0.0
        for layer in self.layers:
            total_loss += layer.get_lattice_regularization_loss()
        return total_loss


# Example usage and testing
if __name__ == "__main__":
    # Test single layer
    layer = LatticeConstrainedLayer(
        input_dim=8, output_dim=4, lattice_type="E8", 
        constraint_strength=0.8, activation="relu"
    )
    
    # Test forward pass
    x = np.random.randn(10, 8)  # Batch of 10 samples
    output = layer.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Regularization loss: {layer.get_lattice_regularization_loss():.6f}")
    
    # Test constraint info
    constraint_info = layer.get_constraint_info()
    print(f"Constraint info: {constraint_info}")
    
    # Test MLP
    mlp = LatticeConstrainedMLP(
        layer_dims=[8, 16, 8, 4], 
        lattice_type="E8", 
        constraint_strength=0.5
    )
    
    mlp_output = mlp.forward(x)
    print(f"MLP output shape: {mlp_output.shape}")
    print(f"MLP total regularization loss: {mlp.get_total_regularization_loss():.6f}")
    
    # Test different lattice types
    z_layer = LatticeConstrainedLayer(
        input_dim=4, output_dim=4, lattice_type="Z", 
        constraint_strength=1.0
    )
    
    z_output = z_layer.forward(np.random.randn(5, 4))
    print(f"Integer lattice output: {z_output}")
    
    print("Lattice-constrained layers test passed!")

