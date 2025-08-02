"""
Interactive 24D Hologram Viewer for lattice structures.

This module implements visualization techniques for high-dimensional lattice
structures, particularly the 24-dimensional Leech lattice, using holographic
projection and interactive exploration methods.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Callable
import json
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class HologramViewer:
    """
    Interactive holographic viewer for high-dimensional lattice structures.
    
    This class provides methods to visualize and interact with lattice
    structures in dimensions higher than 3D using holographic projections
    and dimensional slicing techniques.
    """
    
    def __init__(self, lattice_type: str = "Leech", projection_dim: int = 3):
        """
        Initialize hologram viewer.
        
        Args:
            lattice_type: Type of lattice ("E8", "Leech", "A2", "D4")
            projection_dim: Dimension for holographic projection (2 or 3)
        """
        self.lattice_type = lattice_type
        self.projection_dim = projection_dim
        
        # Set up lattice structure
        self.lattice_basis = self._get_lattice_basis()
        self.source_dimension = len(self.lattice_basis)
        self.lattice_points = self._generate_lattice_points()
        
        # Holographic projection parameters
        self.projection_matrices = self._initialize_projection_matrices()
        self.rotation_matrices = self._initialize_rotation_matrices()
        
        # Visualization state
        self.current_slice = 0
        self.slice_thickness = 1.0
        self.zoom_level = 1.0
        self.rotation_angles = np.zeros(self.source_dimension)
        
        # Color mapping
        self.color_scheme = "dimension_based"
        self.point_sizes = None
        
        logger.info(f"Initialized hologram viewer for {lattice_type} lattice")
        logger.info(f"Source dimension: {self.source_dimension}")
        logger.info(f"Projection dimension: {projection_dim}")
    
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
    
    def _generate_lattice_points(self, max_norm: float = 5.0, max_points: int = 1000) -> np.ndarray:
        """
        Generate lattice points for visualization.
        
        Args:
            max_norm: Maximum norm of points to include
            max_points: Maximum number of points to generate
            
        Returns:
            Array of lattice points
        """
        points = []
        
        if self.lattice_type == "E8":
            # Generate E8 lattice points
            for coords in np.ndindex(*([7] * 8)):  # Limited range
                coord_array = np.array(coords) - 3
                if np.sum(coord_array) % 2 == 0:  # Even sum constraint
                    point = coord_array.astype(float)
                    if np.linalg.norm(point) <= max_norm:
                        points.append(point)
                        if len(points) >= max_points:
                            break
        
        elif self.lattice_type == "Leech":
            # Generate Leech lattice points (simplified)
            for _ in range(max_points):
                coords = np.random.randint(-2, 3, size=24)
                point = coords.astype(float)
                if np.linalg.norm(point) <= max_norm:
                    points.append(point)
        
        elif self.lattice_type == "A2":
            # Generate A2 lattice points
            for i in range(-10, 11):
                for j in range(-10, 11):
                    point = i * self.lattice_basis[0] + j * self.lattice_basis[1]
                    if np.linalg.norm(point) <= max_norm:
                        points.append(point)
                        if len(points) >= max_points:
                            break
                if len(points) >= max_points:
                    break
        
        elif self.lattice_type == "D4":
            # Generate D4 lattice points
            for coords in np.ndindex(*([7] * 4)):
                coord_array = np.array(coords) - 3
                if np.sum(coord_array) % 2 == 0:
                    point = coord_array.astype(float)
                    if np.linalg.norm(point) <= max_norm:
                        points.append(point)
                        if len(points) >= max_points:
                            break
        
        return np.array(points) if points else np.array([]).reshape(0, self.source_dimension)
    
    def _initialize_projection_matrices(self) -> List[np.ndarray]:
        """
        Initialize holographic projection matrices.
        
        Returns:
            List of projection matrices for different views
        """
        projections = []
        
        # Primary orthogonal projections
        for i in range(min(self.source_dimension, 10)):  # Limit number of projections
            projection = np.zeros((self.projection_dim, self.source_dimension))
            
            if self.projection_dim == 2:
                # 2D projections
                projection[0, i] = 1.0
                projection[1, (i + 1) % self.source_dimension] = 1.0
            
            elif self.projection_dim == 3:
                # 3D projections
                projection[0, i] = 1.0
                projection[1, (i + 1) % self.source_dimension] = 1.0
                projection[2, (i + 2) % self.source_dimension] = 1.0
            
            projections.append(projection)
        
        # Add some random projections for holographic effect
        for _ in range(5):
            projection = np.random.randn(self.projection_dim, self.source_dimension)
            # Orthogonalize
            projection = self._gram_schmidt(projection)
            projections.append(projection)
        
        logger.debug(f"Initialized {len(projections)} projection matrices")
        return projections
    
    def _initialize_rotation_matrices(self) -> List[np.ndarray]:
        """
        Initialize rotation matrices for animation.
        
        Returns:
            List of rotation matrices
        """
        rotations = []
        
        # Generate rotation matrices for pairs of dimensions
        for i in range(self.source_dimension):
            for j in range(i + 1, self.source_dimension):
                rotation = np.eye(self.source_dimension)
                # Small rotation in the i-j plane
                angle = 0.1  # Small angle for smooth animation
                rotation[i, i] = np.cos(angle)
                rotation[i, j] = -np.sin(angle)
                rotation[j, i] = np.sin(angle)
                rotation[j, j] = np.cos(angle)
                rotations.append(rotation)
                
                if len(rotations) >= 20:  # Limit number of rotations
                    break
            if len(rotations) >= 20:
                break
        
        return rotations
    
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
            
            for j in range(i):
                projection = np.dot(orthogonal[i], orthogonal[j]) / np.dot(orthogonal[j], orthogonal[j])
                orthogonal[i] -= projection * orthogonal[j]
            
            norm = np.linalg.norm(orthogonal[i])
            if norm > 1e-10:
                orthogonal[i] /= norm
        
        return orthogonal
    
    def project_points(self, projection_index: int = 0) -> np.ndarray:
        """
        Project lattice points using specified projection matrix.
        
        Args:
            projection_index: Index of projection matrix to use
            
        Returns:
            Projected points
        """
        if projection_index >= len(self.projection_matrices):
            projection_index = 0
        
        projection_matrix = self.projection_matrices[projection_index]
        
        # Apply current rotation
        rotated_points = self.lattice_points.copy()
        for i, angle in enumerate(self.rotation_angles[:len(self.rotation_matrices)]):
            if i < len(self.rotation_matrices):
                rotation = self._create_rotation_matrix(i, angle)
                rotated_points = rotated_points @ rotation.T
        
        # Project to lower dimension
        projected = rotated_points @ projection_matrix.T
        
        # Apply zoom
        projected *= self.zoom_level
        
        logger.debug(f"Projected {len(rotated_points)} points using projection {projection_index}")
        return projected
    
    def _create_rotation_matrix(self, rotation_index: int, angle: float) -> np.ndarray:
        """
        Create rotation matrix for given angle.
        
        Args:
            rotation_index: Index of rotation type
            angle: Rotation angle
            
        Returns:
            Rotation matrix
        """
        if rotation_index >= len(self.rotation_matrices):
            return np.eye(self.source_dimension)
        
        # Get base rotation and scale by angle
        base_rotation = self.rotation_matrices[rotation_index]
        
        # Extract rotation plane
        rotation_matrix = np.eye(self.source_dimension)
        
        # Find the two dimensions being rotated
        for i in range(self.source_dimension):
            for j in range(i + 1, self.source_dimension):
                if abs(base_rotation[i, j]) > 1e-6:
                    # Apply rotation in i-j plane
                    rotation_matrix[i, i] = np.cos(angle)
                    rotation_matrix[i, j] = -np.sin(angle)
                    rotation_matrix[j, i] = np.sin(angle)
                    rotation_matrix[j, j] = np.cos(angle)
                    return rotation_matrix
        
        return rotation_matrix
    
    def create_slice(self, slice_dimension: int, slice_value: float, 
                    thickness: float = 1.0) -> np.ndarray:
        """
        Create a slice through the high-dimensional lattice.
        
        Args:
            slice_dimension: Dimension to slice along
            slice_value: Value at which to slice
            thickness: Thickness of the slice
            
        Returns:
            Points within the slice
        """
        if slice_dimension >= self.source_dimension:
            slice_dimension = 0
        
        # Find points within slice
        slice_mask = np.abs(self.lattice_points[:, slice_dimension] - slice_value) <= thickness / 2
        slice_points = self.lattice_points[slice_mask]
        
        logger.debug(f"Created slice with {len(slice_points)} points")
        return slice_points
    
    def animate_rotation(self, steps: int = 100) -> List[np.ndarray]:
        """
        Generate animation frames by rotating through high-dimensional space.
        
        Args:
            steps: Number of animation steps
            
        Returns:
            List of projected point arrays for each frame
        """
        frames = []
        
        for step in range(steps):
            # Update rotation angles
            for i in range(len(self.rotation_angles)):
                self.rotation_angles[i] = 2 * np.pi * step / steps
            
            # Project points for current rotation
            projected = self.project_points(projection_index=0)
            frames.append(projected.copy())
        
        logger.info(f"Generated {len(frames)} animation frames")
        return frames
    
    def compute_point_colors(self, points: np.ndarray) -> np.ndarray:
        """
        Compute colors for points based on their high-dimensional properties.
        
        Args:
            points: High-dimensional points
            
        Returns:
            RGB color array
        """
        colors = np.zeros((len(points), 3))
        
        if self.color_scheme == "dimension_based":
            # Color based on coordinate values in different dimensions
            if self.source_dimension >= 3:
                colors[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min() + 1e-8)
                colors[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min() + 1e-8)
                colors[:, 2] = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-8)
            else:
                colors[:, 0] = 0.5
                colors[:, 1] = 0.5
                colors[:, 2] = 0.8
        
        elif self.color_scheme == "norm_based":
            # Color based on norm
            norms = np.linalg.norm(points, axis=1)
            normalized_norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
            colors[:, 0] = normalized_norms
            colors[:, 1] = 1 - normalized_norms
            colors[:, 2] = 0.5
        
        elif self.color_scheme == "lattice_based":
            # Color based on lattice properties
            if self.lattice_type == "E8":
                # Color based on E8 root properties
                for i, point in enumerate(points):
                    norm_squared = np.dot(point, point)
                    if abs(norm_squared - 2) < 0.1:  # Root vectors
                        colors[i] = [1, 0, 0]  # Red for roots
                    elif norm_squared < 1:
                        colors[i] = [0, 1, 0]  # Green for short vectors
                    else:
                        colors[i] = [0, 0, 1]  # Blue for others
            
            elif self.lattice_type == "Leech":
                # Color based on Leech lattice properties
                for i, point in enumerate(points):
                    norm_squared = np.dot(point, point)
                    if abs(norm_squared - 4) < 0.1:  # Minimal vectors
                        colors[i] = [1, 0, 0]  # Red for minimal vectors
                    elif norm_squared < 2:
                        colors[i] = [0, 1, 0]  # Green for short vectors
                    else:
                        colors[i] = [0, 0, 1]  # Blue for others
        
        # Ensure colors are in [0, 1] range
        colors = np.clip(colors, 0, 1)
        
        return colors
    
    def compute_point_sizes(self, points: np.ndarray) -> np.ndarray:
        """
        Compute point sizes based on their properties.
        
        Args:
            points: High-dimensional points
            
        Returns:
            Array of point sizes
        """
        sizes = np.ones(len(points))
        
        # Size based on distance from origin
        norms = np.linalg.norm(points, axis=1)
        max_norm = norms.max() if len(norms) > 0 else 1
        
        # Larger points for special lattice points
        if self.lattice_type == "E8":
            for i, point in enumerate(points):
                norm_squared = np.dot(point, point)
                if abs(norm_squared - 2) < 0.1:  # Root vectors
                    sizes[i] = 2.0
                elif norm_squared < 0.1:  # Origin
                    sizes[i] = 3.0
        
        elif self.lattice_type == "Leech":
            for i, point in enumerate(points):
                norm_squared = np.dot(point, point)
                if abs(norm_squared - 4) < 0.1:  # Minimal vectors
                    sizes[i] = 2.0
                elif norm_squared < 0.1:  # Origin
                    sizes[i] = 3.0
        
        return sizes
    
    def export_visualization_data(self, filename: str, projection_index: int = 0) -> Dict:
        """
        Export visualization data for external rendering.
        
        Args:
            filename: Output filename
            projection_index: Projection to use
            
        Returns:
            Dictionary with visualization data
        """
        # Project points
        projected_points = self.project_points(projection_index)
        
        # Compute colors and sizes
        colors = self.compute_point_colors(self.lattice_points)
        sizes = self.compute_point_sizes(self.lattice_points)
        
        # Create visualization data
        viz_data = {
            "lattice_type": self.lattice_type,
            "source_dimension": self.source_dimension,
            "projection_dimension": self.projection_dim,
            "points": {
                "original": self.lattice_points.tolist(),
                "projected": projected_points.tolist(),
                "colors": colors.tolist(),
                "sizes": sizes.tolist()
            },
            "projection_matrix": self.projection_matrices[projection_index].tolist(),
            "visualization_params": {
                "zoom_level": self.zoom_level,
                "rotation_angles": self.rotation_angles.tolist(),
                "color_scheme": self.color_scheme
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        logger.info(f"Exported visualization data to {filename}")
        return viz_data
    
    def get_viewer_info(self) -> Dict:
        """
        Get comprehensive information about the hologram viewer.
        
        Returns:
            Dictionary with viewer information
        """
        return {
            "lattice_type": self.lattice_type,
            "source_dimension": self.source_dimension,
            "projection_dimension": self.projection_dim,
            "num_lattice_points": len(self.lattice_points),
            "num_projection_matrices": len(self.projection_matrices),
            "num_rotation_matrices": len(self.rotation_matrices),
            "current_slice": self.current_slice,
            "slice_thickness": self.slice_thickness,
            "zoom_level": self.zoom_level,
            "color_scheme": self.color_scheme
        }


# Example usage and testing
if __name__ == "__main__":
    # Test Leech lattice hologram viewer
    print("Leech Lattice Hologram Viewer:")
    leech_viewer = HologramViewer(lattice_type="Leech", projection_dim=3)
    
    leech_info = leech_viewer.get_viewer_info()
    for key, value in leech_info.items():
        print(f"  {key}: {value}")
    
    # Test projection
    projected_points = leech_viewer.project_points(projection_index=0)
    print(f"  Projected to shape: {projected_points.shape}")
    
    # Test colors and sizes
    colors = leech_viewer.compute_point_colors(leech_viewer.lattice_points)
    sizes = leech_viewer.compute_point_sizes(leech_viewer.lattice_points)
    print(f"  Colors shape: {colors.shape}")
    print(f"  Sizes shape: {sizes.shape}")
    
    # Test slicing
    slice_points = leech_viewer.create_slice(slice_dimension=0, slice_value=0.0, thickness=1.0)
    print(f"  Slice contains {len(slice_points)} points")
    
    # Test E8 hologram viewer
    print("\nE8 Lattice Hologram Viewer:")
    e8_viewer = HologramViewer(lattice_type="E8", projection_dim=2)
    
    e8_info = e8_viewer.get_viewer_info()
    print(f"  Source dimension: {e8_info['source_dimension']}")
    print(f"  Projection dimension: {e8_info['projection_dimension']}")
    print(f"  Number of points: {e8_info['num_lattice_points']}")
    
    # Test animation
    print("\nAnimation Test:")
    animation_frames = e8_viewer.animate_rotation(steps=10)
    print(f"  Generated {len(animation_frames)} animation frames")
    print(f"  Frame shape: {animation_frames[0].shape}")
    
    # Test export
    print("\nExport Test:")
    viz_data = leech_viewer.export_visualization_data("test_hologram.json")
    print(f"  Exported {len(viz_data['points']['original'])} points")
    
    print("Hologram viewer test passed!")

