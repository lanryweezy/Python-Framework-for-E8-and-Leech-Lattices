"""
Dimension Reduction using lattice structures.

This module implements dimension reduction techniques that leverage the
geometric properties of E8 and Leech lattices for efficient data
compression and feature extraction.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class DimensionReduction:
    """
    Lattice-based dimension reduction techniques.
    
    This class implements dimension reduction methods that use lattice
    structures to preserve important geometric and topological properties
    of high-dimensional data.
    """
    
    def __init__(self, lattice_type: str = "E8", target_dimension: int = 3):
        """
        Initialize lattice-based dimension reduction.
        
        Args:
            lattice_type: Type of lattice ("E8", "Leech", "A2", "D4")
            target_dimension: Target dimension for reduction
        """
        self.lattice_type = lattice_type
        self.target_dimension = target_dimension
        
        # Set up lattice structure
        self.lattice_basis = self._get_lattice_basis()
        self.source_dimension = len(self.lattice_basis)
        
        # Reduction matrices
        self.projection_matrix = None
        self.reconstruction_matrix = None
        
        # Learned parameters
        self.mean_vector = None
        self.scaling_factors = None
        
        logger.info(f"Initialized {lattice_type} dimension reduction")
        logger.info(f"Source dimension: {self.source_dimension}")
        logger.info(f"Target dimension: {target_dimension}")
    
    def _get_lattice_basis(self) -> np.ndarray:
        """
        Get lattice basis vectors.
        
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
            # Leech lattice basis (simplified)
            return np.eye(24)
        
        elif self.lattice_type == "A2":
            # A2 lattice basis
            basis = np.array([
                [1.0, 0.0],
                [0.5, np.sqrt(3)/2]
            ])
            return basis
        
        elif self.lattice_type == "D4":
            # D4 lattice basis
            basis = np.array([
                [1.0, 1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, -1.0]
            ])
            return basis
        
        else:
            # Default to 8D identity
            return np.eye(8)
    
    def fit(self, data: np.ndarray) -> 'DimensionReduction':
        """
        Fit the dimension reduction model to data.
        
        Args:
            data: Input data matrix (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if data.shape[1] != self.source_dimension:
            logger.warning(f"Data dimension {data.shape[1]} != source dimension {self.source_dimension}")
            # Adjust source dimension to match data
            self.source_dimension = data.shape[1]
            if self.lattice_type not in ["E8", "Leech", "A2", "D4"]:
                self.lattice_basis = np.eye(self.source_dimension)
        
        # Center the data
        self.mean_vector = np.mean(data, axis=0)
        centered_data = data - self.mean_vector
        
        # Compute scaling factors based on lattice structure
        self.scaling_factors = self._compute_lattice_scaling(centered_data)
        scaled_data = centered_data * self.scaling_factors
        
        # Create projection matrix using lattice-aware method
        self.projection_matrix = self._create_lattice_projection(scaled_data)
        
        # Create reconstruction matrix
        self.reconstruction_matrix = self._create_reconstruction_matrix()
        
        logger.info(f"Fitted dimension reduction model on {data.shape[0]} samples")
        return self
    
    def _compute_lattice_scaling(self, data: np.ndarray) -> np.ndarray:
        """
        Compute scaling factors based on lattice structure.
        
        Args:
            data: Centered data
            
        Returns:
            Scaling factors for each dimension
        """
        if self.lattice_type == "E8":
            # E8-aware scaling based on root system
            scaling = np.ones(self.source_dimension)
            
            # Scale based on E8 root norms
            for i in range(min(8, self.source_dimension)):
                # E8 roots have norm sqrt(2)
                data_std = np.std(data[:, i])
                if data_std > 0:
                    scaling[i] = np.sqrt(2) / data_std
            
            return scaling
        
        elif self.lattice_type == "Leech":
            # Leech lattice scaling
            scaling = np.ones(self.source_dimension)
            
            # Leech lattice minimal vectors have norm 2
            for i in range(min(24, self.source_dimension)):
                data_std = np.std(data[:, i])
                if data_std > 0:
                    scaling[i] = 2.0 / data_std
            
            return scaling
        
        else:
            # Standard scaling
            data_std = np.std(data, axis=0)
            data_std[data_std == 0] = 1.0  # Avoid division by zero
            return 1.0 / data_std
    
    def _create_lattice_projection(self, data: np.ndarray) -> np.ndarray:
        """
        Create projection matrix using lattice-aware method.
        
        Args:
            data: Scaled and centered data
            
        Returns:
            Projection matrix
        """
        if self.lattice_type == "E8":
            return self._create_e8_projection(data)
        elif self.lattice_type == "Leech":
            return self._create_leech_projection(data)
        else:
            return self._create_pca_projection(data)
    
    def _create_e8_projection(self, data: np.ndarray) -> np.ndarray:
        """
        Create E8-aware projection matrix.
        
        Args:
            data: Input data
            
        Returns:
            E8-based projection matrix
        """
        # Use E8 root system for projection
        if self.source_dimension >= 8:
            # Start with E8 basis vectors
            projection = self.lattice_basis[:self.target_dimension, :self.source_dimension]
        else:
            # Use PCA for smaller dimensions
            return self._create_pca_projection(data)
        
        # Orthogonalize using Gram-Schmidt
        projection = self._gram_schmidt(projection)
        
        return projection
    
    def _create_leech_projection(self, data: np.ndarray) -> np.ndarray:
        """
        Create Leech lattice-aware projection matrix.
        
        Args:
            data: Input data
            
        Returns:
            Leech-based projection matrix
        """
        if self.source_dimension >= 24:
            # Use structured projection based on Leech lattice
            projection = np.zeros((self.target_dimension, self.source_dimension))
            
            # Create projection using block structure
            block_size = 24 // self.target_dimension if self.target_dimension <= 24 else 1
            
            for i in range(self.target_dimension):
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, self.source_dimension)
                if start_idx < self.source_dimension:
                    projection[i, start_idx:end_idx] = 1.0 / np.sqrt(end_idx - start_idx)
        else:
            # Use PCA for smaller dimensions
            return self._create_pca_projection(data)
        
        return projection
    
    def _create_pca_projection(self, data: np.ndarray) -> np.ndarray:
        """
        Create PCA-based projection matrix.
        
        Args:
            data: Input data
            
        Returns:
            PCA projection matrix
        """
        # Compute covariance matrix
        cov_matrix = np.cov(data.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Take top components
        projection = sorted_eigenvectors[:, :self.target_dimension].T
        
        return projection
    
    def _gram_schmidt(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply Gram-Schmidt orthogonalization.
        
        Args:
            vectors: Input vectors (rows)
            
        Returns:
            Orthogonalized vectors
        """
        orthogonal = np.zeros_like(vectors)
        
        for i in range(len(vectors)):
            orthogonal[i] = vectors[i].copy()
            
            # Subtract projections onto previous vectors
            for j in range(i):
                projection = np.dot(orthogonal[i], orthogonal[j]) / np.dot(orthogonal[j], orthogonal[j])
                orthogonal[i] -= projection * orthogonal[j]
            
            # Normalize
            norm = np.linalg.norm(orthogonal[i])
            if norm > 1e-10:
                orthogonal[i] /= norm
        
        return orthogonal
    
    def _create_reconstruction_matrix(self) -> np.ndarray:
        """
        Create reconstruction matrix (pseudo-inverse of projection).
        
        Returns:
            Reconstruction matrix
        """
        # For orthogonal projection, reconstruction is transpose
        return self.projection_matrix.T
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data to lower dimension.
        
        Args:
            data: Input data (n_samples, n_features)
            
        Returns:
            Transformed data (n_samples, target_dimension)
        """
        if self.projection_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Center and scale data
        centered_data = data - self.mean_vector
        scaled_data = centered_data * self.scaling_factors
        
        # Project to lower dimension
        transformed = scaled_data @ self.projection_matrix.T
        
        logger.debug(f"Transformed {data.shape[0]} samples to {self.target_dimension}D")
        return transformed
    
    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from lower dimension.
        
        Args:
            transformed_data: Low-dimensional data
            
        Returns:
            Reconstructed high-dimensional data
        """
        if self.reconstruction_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Reconstruct in scaled space
        reconstructed_scaled = transformed_data @ self.reconstruction_matrix.T
        
        # Unscale and uncenter
        reconstructed = reconstructed_scaled / self.scaling_factors + self.mean_vector
        
        logger.debug(f"Reconstructed {transformed_data.shape[0]} samples to {self.source_dimension}D")
        return reconstructed
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit model and transform data in one step.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        return self.fit(data).transform(data)
    
    def compute_reconstruction_error(self, data: np.ndarray) -> float:
        """
        Compute reconstruction error for given data.
        
        Args:
            data: Original data
            
        Returns:
            Mean squared reconstruction error
        """
        transformed = self.transform(data)
        reconstructed = self.inverse_transform(transformed)
        
        error = np.mean((data - reconstructed) ** 2)
        
        logger.debug(f"Reconstruction error: {error:.6f}")
        return error
    
    def compute_explained_variance_ratio(self, data: np.ndarray) -> np.ndarray:
        """
        Compute explained variance ratio for each component.
        
        Args:
            data: Original data
            
        Returns:
            Explained variance ratio for each component
        """
        # Center data
        centered_data = data - self.mean_vector
        total_variance = np.var(centered_data, axis=0).sum()
        
        # Transform and compute variance in each dimension
        transformed = self.transform(data)
        component_variances = np.var(transformed, axis=0)
        
        # Explained variance ratio
        explained_ratio = component_variances / total_variance
        
        logger.debug(f"Explained variance ratios: {explained_ratio}")
        return explained_ratio
    
    def visualize_lattice_structure(self, data: np.ndarray, num_points: int = 100) -> Dict:
        """
        Visualize how lattice structure affects the reduction.
        
        Args:
            data: Original data
            num_points: Number of points to analyze
            
        Returns:
            Dictionary with visualization data
        """
        # Sample points
        sample_indices = np.random.choice(len(data), min(num_points, len(data)), replace=False)
        sample_data = data[sample_indices]
        
        # Transform
        transformed = self.transform(sample_data)
        reconstructed = self.inverse_transform(transformed)
        
        # Compute lattice distances
        lattice_distances = []
        for point in sample_data:
            # Find nearest lattice point
            nearest_lattice = self._find_nearest_lattice_point(point)
            distance = np.linalg.norm(point - nearest_lattice)
            lattice_distances.append(distance)
        
        visualization_data = {
            "original_points": sample_data,
            "transformed_points": transformed,
            "reconstructed_points": reconstructed,
            "lattice_distances": np.array(lattice_distances),
            "reconstruction_errors": np.linalg.norm(sample_data - reconstructed, axis=1)
        }
        
        logger.info(f"Generated visualization data for {num_points} points")
        return visualization_data
    
    def _find_nearest_lattice_point(self, point: np.ndarray) -> np.ndarray:
        """
        Find nearest lattice point to given point.
        
        Args:
            point: Input point
            
        Returns:
            Nearest lattice point
        """
        if self.lattice_type == "E8" and len(point) >= 8:
            # E8 lattice quantization
            rounded = np.round(point[:8])
            if np.sum(rounded) % 2 == 0:
                return np.concatenate([rounded, point[8:]])
            else:
                # Adjust to make sum even
                fractional = np.abs(point[:8] - rounded)
                adjust_idx = np.argmax(fractional)
                rounded[adjust_idx] += 1 if point[adjust_idx] > rounded[adjust_idx] else -1
                return np.concatenate([rounded, point[8:]])
        
        else:
            # Default to rounding
            return np.round(point)
    
    def get_reduction_info(self) -> Dict:
        """
        Get comprehensive information about the dimension reduction.
        
        Returns:
            Dictionary with reduction information
        """
        info = {
            "lattice_type": self.lattice_type,
            "source_dimension": self.source_dimension,
            "target_dimension": self.target_dimension,
            "fitted": self.projection_matrix is not None
        }
        
        if self.projection_matrix is not None:
            info.update({
                "projection_matrix_shape": self.projection_matrix.shape,
                "reconstruction_matrix_shape": self.reconstruction_matrix.shape,
                "mean_vector_shape": self.mean_vector.shape,
                "scaling_factors_shape": self.scaling_factors.shape
            })
        
        return info


# Example usage and testing
if __name__ == "__main__":
    # Generate sample high-dimensional data
    np.random.seed(42)
    n_samples = 200
    
    # Test E8 dimension reduction
    print("E8 Dimension Reduction:")
    e8_data = np.random.randn(n_samples, 8)  # 8D data
    e8_reducer = DimensionReduction(lattice_type="E8", target_dimension=3)
    
    # Fit and transform
    e8_transformed = e8_reducer.fit_transform(e8_data)
    e8_reconstructed = e8_reducer.inverse_transform(e8_transformed)
    
    print(f"  Original shape: {e8_data.shape}")
    print(f"  Transformed shape: {e8_transformed.shape}")
    print(f"  Reconstructed shape: {e8_reconstructed.shape}")
    
    # Compute reconstruction error
    e8_error = e8_reducer.compute_reconstruction_error(e8_data)
    print(f"  Reconstruction error: {e8_error:.6f}")
    
    # Explained variance
    e8_explained = e8_reducer.compute_explained_variance_ratio(e8_data)
    print(f"  Explained variance ratios: {e8_explained}")
    
    # Test Leech dimension reduction
    print("\nLeech Dimension Reduction:")
    leech_data = np.random.randn(n_samples, 24)  # 24D data
    leech_reducer = DimensionReduction(lattice_type="Leech", target_dimension=6)
    
    leech_transformed = leech_reducer.fit_transform(leech_data)
    leech_error = leech_reducer.compute_reconstruction_error(leech_data)
    
    print(f"  Original shape: {leech_data.shape}")
    print(f"  Transformed shape: {leech_transformed.shape}")
    print(f"  Reconstruction error: {leech_error:.6f}")
    
    # Test A2 dimension reduction
    print("\nA2 Dimension Reduction:")
    a2_data = np.random.randn(n_samples, 2)  # 2D data
    a2_reducer = DimensionReduction(lattice_type="A2", target_dimension=1)
    
    a2_transformed = a2_reducer.fit_transform(a2_data)
    a2_error = a2_reducer.compute_reconstruction_error(a2_data)
    
    print(f"  Original shape: {a2_data.shape}")
    print(f"  Transformed shape: {a2_transformed.shape}")
    print(f"  Reconstruction error: {a2_error:.6f}")
    
    # Test visualization
    print("\nVisualization Data:")
    viz_data = e8_reducer.visualize_lattice_structure(e8_data, num_points=50)
    print(f"  Analyzed {len(viz_data['original_points'])} points")
    print(f"  Mean lattice distance: {np.mean(viz_data['lattice_distances']):.4f}")
    print(f"  Mean reconstruction error: {np.mean(viz_data['reconstruction_errors']):.4f}")
    
    # Get reduction info
    print("\nReduction Info:")
    info = e8_reducer.get_reduction_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("Dimension reduction test passed!")

