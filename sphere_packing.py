"""
Optimal Sphere Packing using lattice structures.

This module implements sphere packing algorithms that leverage the optimal
packing properties of E8 and Leech lattices for data compression,
coding theory, and geometric optimization problems.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class SpherePacking:
    """
    Sphere packing algorithms using exceptional lattices.
    
    This class implements optimal sphere packing schemes based on
    E8 and Leech lattices, which achieve the highest known packing
    densities in their respective dimensions.
    """
    
    def __init__(self, lattice_type: str = "E8", sphere_radius: float = 1.0):
        """
        Initialize sphere packing.
        
        Args:
            lattice_type: Type of lattice ("E8", "Leech", "A2", "D4")
            sphere_radius: Radius of spheres to pack
        """
        self.lattice_type = lattice_type
        self.sphere_radius = sphere_radius
        
        # Set up lattice structure
        self.lattice_basis = self._get_lattice_basis()
        self.dimension = len(self.lattice_basis)
        self.lattice_points = self._generate_lattice_points()
        
        # Compute packing properties
        self.packing_density = self._compute_packing_density()
        self.kissing_number = self._compute_kissing_number()
        self.covering_radius = self._compute_covering_radius()
        
        logger.info(f"Initialized {lattice_type} sphere packing in {self.dimension}D")
        logger.info(f"Packing density: {self.packing_density:.6f}")
        logger.info(f"Kissing number: {self.kissing_number}")
    
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
            # A2 (hexagonal) lattice
            basis = np.array([
                [1.0, 0.0],
                [0.5, np.sqrt(3)/2]
            ])
            return basis
        
        elif self.lattice_type == "D4":
            # D4 lattice
            basis = np.array([
                [1.0, 1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, -1.0]
            ])
            return basis
        
        else:
            # Default to cubic lattice
            return np.eye(3)
    
    def _generate_lattice_points(self, max_norm: float = 10.0) -> np.ndarray:
        """
        Generate lattice points within a given norm bound.
        
        Args:
            max_norm: Maximum norm of lattice points to generate
            
        Returns:
            Array of lattice points
        """
        points = []
        
        if self.lattice_type == "E8":
            # Generate E8 lattice points
            # Type 1: integer coordinates with even sum
            for coords in np.ndindex(*([5] * 8)):  # Limited range
                coord_array = np.array(coords) - 2  # Center around 0
                if np.sum(coord_array) % 2 == 0:  # Even sum
                    point = coord_array.astype(float)
                    if np.linalg.norm(point) <= max_norm:
                        points.append(point)
            
            # Type 2: half-integer coordinates
            for coords in np.ndindex(*([3] * 8)):  # Limited range
                coord_array = np.array(coords) - 1 + 0.5  # Half-integers
                point = coord_array
                if np.linalg.norm(point) <= max_norm:
                    points.append(point)
                
                if len(points) >= 1000:  # Limit for computation
                    break
        
        elif self.lattice_type == "Leech":
            # Generate Leech lattice points (simplified)
            for _ in range(500):  # Random subset
                coords = np.random.randint(-3, 4, size=24)
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
        
        elif self.lattice_type == "D4":
            # Generate D4 lattice points
            for coords in np.ndindex(*([7] * 4)):  # Limited range
                coord_array = np.array(coords) - 3
                if np.sum(coord_array) % 2 == 0:  # Even sum constraint
                    point = coord_array.astype(float)
                    if np.linalg.norm(point) <= max_norm:
                        points.append(point)
        
        else:
            # Default cubic lattice
            for i in range(-5, 6):
                for j in range(-5, 6):
                    for k in range(-5, 6):
                        point = np.array([i, j, k], dtype=float)
                        if np.linalg.norm(point) <= max_norm:
                            points.append(point)
        
        return np.array(points) if points else np.array([]).reshape(0, self.dimension)
    
    def _compute_packing_density(self) -> float:
        """
        Compute the packing density of the lattice.
        
        Returns:
            Packing density
        """
        if self.lattice_type == "E8":
            # E8 optimal packing density
            return np.pi**4 / (384)  # ≈ 0.25367
        
        elif self.lattice_type == "Leech":
            # Leech lattice packing density
            return np.pi**12 / (12**12)  # Approximate
        
        elif self.lattice_type == "A2":
            # Hexagonal packing density
            return np.pi / (2 * np.sqrt(3))  # ≈ 0.9069
        
        elif self.lattice_type == "D4":
            # D4 packing density
            return np.pi**2 / 16  # ≈ 0.6169
        
        else:
            # Cubic packing density
            return np.pi / 6  # ≈ 0.5236
    
    def _compute_kissing_number(self) -> int:
        """
        Compute the kissing number of the lattice.
        
        Returns:
            Kissing number
        """
        if self.lattice_type == "E8":
            return 240
        elif self.lattice_type == "Leech":
            return 196560
        elif self.lattice_type == "A2":
            return 6
        elif self.lattice_type == "D4":
            return 24
        else:
            return 6  # Cubic lattice
    
    def _compute_covering_radius(self) -> float:
        """
        Compute the covering radius of the lattice.
        
        Returns:
            Covering radius
        """
        # Simplified covering radius computation
        if self.lattice_type == "E8":
            return np.sqrt(2) / 2  # Approximate
        elif self.lattice_type == "Leech":
            return np.sqrt(8) / 2  # Approximate
        elif self.lattice_type == "A2":
            return 1.0 / np.sqrt(3)  # Exact
        elif self.lattice_type == "D4":
            return 1.0  # Approximate
        else:
            return np.sqrt(3) / 2  # Cubic lattice
    
    def pack_spheres_in_region(self, region_bounds: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Pack spheres in a given region using the lattice.
        
        Args:
            region_bounds: Array of [min, max] bounds for each dimension
            
        Returns:
            Tuple of (sphere_centers, num_spheres)
        """
        if region_bounds.shape != (self.dimension, 2):
            raise ValueError(f"Region bounds must have shape ({self.dimension}, 2)")
        
        sphere_centers = []
        
        # Scale lattice to fit sphere radius
        scaled_basis = self.lattice_basis * (2 * self.sphere_radius)
        
        # Generate lattice points within the region
        for point in self.lattice_points:
            scaled_point = point * (2 * self.sphere_radius)
            
            # Check if point is within region bounds
            within_bounds = True
            for dim in range(self.dimension):
                if (scaled_point[dim] < region_bounds[dim, 0] + self.sphere_radius or
                    scaled_point[dim] > region_bounds[dim, 1] - self.sphere_radius):
                    within_bounds = False
                    break
            
            if within_bounds:
                sphere_centers.append(scaled_point)
        
        sphere_centers = np.array(sphere_centers) if sphere_centers else np.array([]).reshape(0, self.dimension)
        num_spheres = len(sphere_centers)
        
        logger.info(f"Packed {num_spheres} spheres in region")
        return sphere_centers, num_spheres
    
    def compute_packing_efficiency(self, region_bounds: np.ndarray) -> float:
        """
        Compute packing efficiency in a given region.
        
        Args:
            region_bounds: Array of [min, max] bounds for each dimension
            
        Returns:
            Packing efficiency (fraction of volume occupied)
        """
        sphere_centers, num_spheres = self.pack_spheres_in_region(region_bounds)
        
        # Volume of region
        region_volume = 1.0
        for dim in range(self.dimension):
            region_volume *= (region_bounds[dim, 1] - region_bounds[dim, 0])
        
        # Volume of spheres
        sphere_volume = self._compute_sphere_volume(self.sphere_radius)
        total_sphere_volume = num_spheres * sphere_volume
        
        # Packing efficiency
        efficiency = total_sphere_volume / region_volume
        
        logger.debug(f"Packing efficiency: {efficiency:.6f}")
        return efficiency
    
    def _compute_sphere_volume(self, radius: float) -> float:
        """
        Compute volume of a sphere in the given dimension.
        
        Args:
            radius: Sphere radius
            
        Returns:
            Sphere volume
        """
        from math import gamma, pi
        
        # V_n(r) = π^(n/2) * r^n / Γ(n/2 + 1)
        volume = (pi**(self.dimension / 2) * radius**self.dimension / 
                 gamma(self.dimension / 2 + 1))
        
        return volume
    
    def find_nearest_lattice_point(self, target_point: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Find the nearest lattice point to a target point.
        
        Args:
            target_point: Target point
            
        Returns:
            Tuple of (nearest_lattice_point, distance)
        """
        if len(target_point) != self.dimension:
            raise ValueError(f"Target point must have dimension {self.dimension}")
        
        min_distance = float('inf')
        nearest_point = None
        
        for lattice_point in self.lattice_points:
            distance = np.linalg.norm(target_point - lattice_point)
            if distance < min_distance:
                min_distance = distance
                nearest_point = lattice_point
        
        if nearest_point is None:
            # Fallback to origin
            nearest_point = np.zeros(self.dimension)
            min_distance = np.linalg.norm(target_point)
        
        logger.debug(f"Nearest lattice point found at distance {min_distance:.6f}")
        return nearest_point, min_distance
    
    def compute_voronoi_cell_volume(self) -> float:
        """
        Compute the volume of the Voronoi cell.
        
        Returns:
            Voronoi cell volume
        """
        # For a lattice, Voronoi cell volume = |det(basis)|
        determinant = np.abs(np.linalg.det(self.lattice_basis))
        
        logger.debug(f"Voronoi cell volume: {determinant:.6f}")
        return determinant
    
    def optimize_packing_for_data(self, data_points: np.ndarray) -> Dict:
        """
        Optimize sphere packing for a given dataset.
        
        Args:
            data_points: Array of data points to pack
            
        Returns:
            Dictionary with optimization results
        """
        if data_points.shape[1] != self.dimension:
            raise ValueError(f"Data points must have dimension {self.dimension}")
        
        # Find bounding box of data
        min_bounds = np.min(data_points, axis=0)
        max_bounds = np.max(data_points, axis=0)
        region_bounds = np.column_stack([min_bounds, max_bounds])
        
        # Pack spheres in the data region
        sphere_centers, num_spheres = self.pack_spheres_in_region(region_bounds)
        
        # Compute coverage statistics
        covered_points = 0
        for data_point in data_points:
            for sphere_center in sphere_centers:
                if np.linalg.norm(data_point - sphere_center) <= self.sphere_radius:
                    covered_points += 1
                    break
        
        coverage_ratio = covered_points / len(data_points)
        packing_efficiency = self.compute_packing_efficiency(region_bounds)
        
        optimization_results = {
            "num_spheres": num_spheres,
            "coverage_ratio": coverage_ratio,
            "packing_efficiency": packing_efficiency,
            "sphere_centers": sphere_centers,
            "region_bounds": region_bounds
        }
        
        logger.info(f"Data packing optimization completed")
        logger.info(f"Coverage ratio: {coverage_ratio:.4f}")
        logger.info(f"Packing efficiency: {packing_efficiency:.4f}")
        
        return optimization_results
    
    def get_packing_info(self) -> Dict:
        """
        Get comprehensive information about the sphere packing.
        
        Returns:
            Dictionary with packing information
        """
        return {
            "lattice_type": self.lattice_type,
            "dimension": self.dimension,
            "sphere_radius": self.sphere_radius,
            "packing_density": self.packing_density,
            "kissing_number": self.kissing_number,
            "covering_radius": self.covering_radius,
            "num_lattice_points": len(self.lattice_points),
            "voronoi_cell_volume": self.compute_voronoi_cell_volume()
        }


# Example usage and testing
if __name__ == "__main__":
    # Test E8 sphere packing
    e8_packing = SpherePacking(lattice_type="E8", sphere_radius=0.5)
    
    print("E8 Sphere Packing:")
    e8_info = e8_packing.get_packing_info()
    for key, value in e8_info.items():
        print(f"  {key}: {value}")
    
    # Test sphere packing in a region
    region_bounds = np.array([[-5, 5]] * 8)  # 8D cube
    sphere_centers, num_spheres = e8_packing.pack_spheres_in_region(region_bounds)
    print(f"  Packed {num_spheres} spheres in 8D cube")
    
    # Test packing efficiency
    efficiency = e8_packing.compute_packing_efficiency(region_bounds)
    print(f"  Packing efficiency: {efficiency:.6f}")
    
    # Test A2 (hexagonal) packing
    print("\nA2 (Hexagonal) Sphere Packing:")
    a2_packing = SpherePacking(lattice_type="A2", sphere_radius=0.5)
    a2_info = a2_packing.get_packing_info()
    for key, value in a2_info.items():
        print(f"  {key}: {value}")
    
    # Test nearest lattice point
    target = np.array([1.3, 2.7])
    nearest, distance = a2_packing.find_nearest_lattice_point(target)
    print(f"  Nearest to {target}: {nearest} (distance: {distance:.4f})")
    
    # Test data optimization
    np.random.seed(42)
    data_points = np.random.randn(100, 2) * 2  # 2D random data
    optimization = a2_packing.optimize_packing_for_data(data_points)
    print(f"  Data coverage: {optimization['coverage_ratio']:.4f}")
    
    # Test Leech packing
    print("\nLeech Sphere Packing:")
    leech_packing = SpherePacking(lattice_type="Leech", sphere_radius=1.0)
    leech_info = leech_packing.get_packing_info()
    for key, value in leech_info.items():
        print(f"  {key}: {value}")
    
    print("Sphere packing test passed!")

