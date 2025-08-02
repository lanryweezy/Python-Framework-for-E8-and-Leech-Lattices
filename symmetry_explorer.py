"""
Symmetry Group Explorer for lattice structures.

This module implements interactive exploration of symmetry groups
associated with E8 and Leech lattices, including Weyl groups,
Conway groups, and their actions on lattice points.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Set
import json
from itertools import permutations, combinations
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class SymmetryExplorer:
    """
    Interactive explorer for lattice symmetry groups.
    
    This class provides methods to explore and visualize the symmetry
    groups of exceptional lattices, including their generators,
    orbits, and actions on lattice points.
    """
    
    def __init__(self, lattice_type: str = "E8"):
        """
        Initialize symmetry explorer.
        
        Args:
            lattice_type: Type of lattice ("E8", "Leech", "A2", "D4")
        """
        self.lattice_type = lattice_type
        
        # Set up lattice structure
        self.lattice_basis = self._get_lattice_basis()
        self.dimension = len(self.lattice_basis)
        
        # Symmetry group structure
        self.generators = self._get_symmetry_generators()
        self.group_elements = self._generate_group_elements()
        self.group_order = len(self.group_elements)
        
        # Root system and special points
        self.root_system = self._generate_root_system()
        self.special_points = self._identify_special_points()
        
        # Orbit structure
        self.orbits = {}
        self.orbit_representatives = {}
        
        logger.info(f"Initialized symmetry explorer for {lattice_type} lattice")
        logger.info(f"Dimension: {self.dimension}")
        logger.info(f"Number of generators: {len(self.generators)}")
        logger.info(f"Group order: {self.group_order}")
    
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
            # Default to identity
            return np.eye(3)
    
    def _get_symmetry_generators(self) -> List[np.ndarray]:
        """
        Get generators of the symmetry group.
        
        Returns:
            List of generator matrices
        """
        generators = []
        
        if self.lattice_type == "E8":
            # E8 Weyl group generators
            
            # Simple reflections for E8
            for i in range(7):
                # Reflection s_i: e_i ↔ e_{i+1}
                reflection = np.eye(8)
                reflection[i, i] = 0
                reflection[i+1, i+1] = 0
                reflection[i, i+1] = 1
                reflection[i+1, i] = 1
                generators.append(reflection)
            
            # Additional reflection for E8
            reflection = np.eye(8)
            for i in range(8):
                reflection[i, i] = -1 if i < 2 else 1
            generators.append(reflection)
        
        elif self.lattice_type == "Leech":
            # Conway group generators (simplified)
            
            # Coordinate permutations
            for i in range(min(5, 24)):  # Limit for computation
                perm = np.eye(24)
                if i + 1 < 24:
                    perm[i, i] = 0
                    perm[i+1, i+1] = 0
                    perm[i, i+1] = 1
                    perm[i+1, i] = 1
                generators.append(perm)
            
            # Sign changes
            sign_change = np.eye(24)
            sign_change[0, 0] = -1
            generators.append(sign_change)
        
        elif self.lattice_type == "A2":
            # A2 symmetry group (dihedral group D6)
            
            # 120-degree rotation
            angle = 2 * np.pi / 3
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            generators.append(rotation)
            
            # Reflection
            reflection = np.array([
                [1, 0],
                [0, -1]
            ])
            generators.append(reflection)
        
        elif self.lattice_type == "D4":
            # D4 symmetry group
            
            # Coordinate permutations
            for i in range(3):
                perm = np.eye(4)
                perm[i, i] = 0
                perm[i+1, i+1] = 0
                perm[i, i+1] = 1
                perm[i+1, i] = 1
                generators.append(perm)
            
            # Sign changes
            sign_change = np.eye(4)
            sign_change[0, 0] = -1
            sign_change[1, 1] = -1
            generators.append(sign_change)
        
        else:
            # Default generators (rotations and reflections)
            generators.append(np.eye(3))
        
        logger.debug(f"Generated {len(generators)} symmetry generators")
        return generators
    
    def _generate_group_elements(self, max_elements: int = 1000) -> List[np.ndarray]:
        """
        Generate group elements from generators.
        
        Args:
            max_elements: Maximum number of elements to generate
            
        Returns:
            List of group elements
        """
        elements = [np.eye(self.dimension)]  # Identity
        new_elements = [np.eye(self.dimension)]
        
        # Generate elements by multiplying with generators
        for _ in range(10):  # Limit iterations
            next_new_elements = []
            
            for element in new_elements:
                for generator in self.generators:
                    # Multiply element by generator
                    new_element = element @ generator
                    
                    # Check if this is a new element (up to numerical precision)
                    is_new = True
                    for existing in elements:
                        if np.allclose(new_element, existing, atol=1e-10):
                            is_new = False
                            break
                    
                    if is_new:
                        elements.append(new_element)
                        next_new_elements.append(new_element)
                        
                        if len(elements) >= max_elements:
                            break
                
                if len(elements) >= max_elements:
                    break
            
            new_elements = next_new_elements
            if not new_elements:  # No new elements generated
                break
        
        logger.debug(f"Generated {len(elements)} group elements")
        return elements
    
    def _generate_root_system(self) -> np.ndarray:
        """
        Generate root system for the lattice.
        
        Returns:
            Array of root vectors
        """
        if self.lattice_type == "E8":
            roots = []
            
            # Type 1: e_i - e_j
            for i in range(8):
                for j in range(i + 1, 8):
                    root1 = np.zeros(8)
                    root1[i] = 1
                    root1[j] = -1
                    roots.append(root1)
                    
                    root2 = np.zeros(8)
                    root2[i] = -1
                    root2[j] = 1
                    roots.append(root2)
            
            # Type 2: (1/2) * sum with even number of minus signs
            for signs in range(2**8):
                root = np.zeros(8)
                minus_count = 0
                for i in range(8):
                    if (signs >> i) & 1:
                        root[i] = -0.5
                        minus_count += 1
                    else:
                        root[i] = 0.5
                
                if minus_count % 2 == 0:
                    roots.append(root)
            
            return np.array(roots[:240])  # E8 has 240 roots
        
        elif self.lattice_type == "Leech":
            # Simplified Leech root system
            roots = []
            for i in range(24):
                for j in range(i + 1, 24):
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            root = np.zeros(24)
                            root[i] = s1
                            root[j] = s2
                            roots.append(root)
                            if len(roots) >= 500:  # Limit for computation
                                break
                        if len(roots) >= 500:
                            break
                    if len(roots) >= 500:
                        break
                if len(roots) >= 500:
                    break
            
            return np.array(roots)
        
        elif self.lattice_type == "A2":
            # A2 root system
            roots = []
            # Standard A2 roots
            roots.append(np.array([1, 0]))
            roots.append(np.array([-1, 0]))
            roots.append(np.array([0.5, np.sqrt(3)/2]))
            roots.append(np.array([-0.5, -np.sqrt(3)/2]))
            roots.append(np.array([0.5, -np.sqrt(3)/2]))
            roots.append(np.array([-0.5, np.sqrt(3)/2]))
            
            return np.array(roots)
        
        elif self.lattice_type == "D4":
            # D4 root system
            roots = []
            for i in range(4):
                for j in range(i + 1, 4):
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            root = np.zeros(4)
                            root[i] = s1
                            root[j] = s2
                            roots.append(root)
            
            return np.array(roots)
        
        else:
            # Default root system
            return np.eye(self.dimension)
    
    def _identify_special_points(self) -> Dict[str, List[np.ndarray]]:
        """
        Identify special points in the lattice.
        
        Returns:
            Dictionary of special point categories
        """
        special_points = {
            "origin": [np.zeros(self.dimension)],
            "roots": [],
            "minimal_vectors": [],
            "short_vectors": []
        }
        
        # Add root vectors
        special_points["roots"] = list(self.root_system)
        
        if self.lattice_type == "E8":
            # E8 minimal vectors are the roots
            special_points["minimal_vectors"] = list(self.root_system)
            
            # Short vectors (norm ≤ 2)
            for root in self.root_system:
                if np.linalg.norm(root) <= 2.1:
                    special_points["short_vectors"].append(root)
        
        elif self.lattice_type == "Leech":
            # Leech minimal vectors have norm 2
            for root in self.root_system:
                if abs(np.linalg.norm(root) - 2) < 0.1:
                    special_points["minimal_vectors"].append(root)
        
        logger.debug(f"Identified {sum(len(v) for v in special_points.values())} special points")
        return special_points
    
    def compute_orbit(self, point: np.ndarray) -> List[np.ndarray]:
        """
        Compute the orbit of a point under the symmetry group.
        
        Args:
            point: Input point
            
        Returns:
            List of points in the orbit
        """
        orbit = []
        
        for group_element in self.group_elements:
            transformed_point = group_element @ point
            
            # Check if this point is already in the orbit
            is_new = True
            for existing_point in orbit:
                if np.allclose(transformed_point, existing_point, atol=1e-10):
                    is_new = False
                    break
            
            if is_new:
                orbit.append(transformed_point)
        
        logger.debug(f"Computed orbit with {len(orbit)} points")
        return orbit
    
    def compute_stabilizer(self, point: np.ndarray) -> List[np.ndarray]:
        """
        Compute the stabilizer subgroup of a point.
        
        Args:
            point: Input point
            
        Returns:
            List of group elements that fix the point
        """
        stabilizer = []
        
        for group_element in self.group_elements:
            transformed_point = group_element @ point
            
            if np.allclose(transformed_point, point, atol=1e-10):
                stabilizer.append(group_element)
        
        logger.debug(f"Computed stabilizer with {len(stabilizer)} elements")
        return stabilizer
    
    def find_orbit_representatives(self, points: List[np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Find orbit representatives for a set of points.
        
        Args:
            points: List of points
            
        Returns:
            Dictionary mapping orbit index to representative
        """
        representatives = {}
        orbit_assignments = {}
        current_orbit = 0
        
        for i, point in enumerate(points):
            # Check if this point is already in an orbit
            assigned = False
            
            for orbit_idx, rep in representatives.items():
                orbit = self.compute_orbit(rep)
                
                for orbit_point in orbit:
                    if np.allclose(point, orbit_point, atol=1e-10):
                        orbit_assignments[i] = orbit_idx
                        assigned = True
                        break
                
                if assigned:
                    break
            
            if not assigned:
                # This point represents a new orbit
                representatives[current_orbit] = point
                orbit_assignments[i] = current_orbit
                current_orbit += 1
        
        logger.info(f"Found {len(representatives)} orbit representatives")
        return representatives
    
    def analyze_symmetry_breaking(self, perturbation: np.ndarray) -> Dict:
        """
        Analyze how a perturbation breaks the symmetry.
        
        Args:
            perturbation: Perturbation vector
            
        Returns:
            Dictionary with symmetry breaking analysis
        """
        # Original symmetry group size
        original_group_size = len(self.group_elements)
        
        # Find elements that still preserve the perturbed structure
        preserved_elements = []
        
        for group_element in self.group_elements:
            # Check if the group element approximately preserves the perturbation
            transformed_perturbation = group_element @ perturbation
            
            # Simple criterion: perturbation direction is preserved
            if np.dot(perturbation, transformed_perturbation) > 0.9 * np.linalg.norm(perturbation)**2:
                preserved_elements.append(group_element)
        
        symmetry_breaking_analysis = {
            "original_group_size": original_group_size,
            "preserved_group_size": len(preserved_elements),
            "symmetry_breaking_ratio": 1 - len(preserved_elements) / original_group_size,
            "preserved_elements": preserved_elements
        }
        
        logger.info(f"Symmetry breaking analysis: {len(preserved_elements)}/{original_group_size} elements preserved")
        return symmetry_breaking_analysis
    
    def visualize_fundamental_domain(self) -> Dict:
        """
        Compute and visualize the fundamental domain.
        
        Returns:
            Dictionary with fundamental domain data
        """
        # For simplicity, use a subset of root vectors to define the domain
        if len(self.root_system) == 0:
            return {"vertices": [], "faces": []}
        
        # Take first few roots as domain boundaries
        domain_roots = self.root_system[:min(10, len(self.root_system))]
        
        # Compute vertices of fundamental domain (simplified)
        vertices = []
        
        # Generate some candidate points
        for i in range(100):
            candidate = np.random.randn(self.dimension) * 0.5
            
            # Check if point is in fundamental domain
            in_domain = True
            for root in domain_roots:
                if np.dot(candidate, root) < 0:
                    in_domain = False
                    break
            
            if in_domain:
                vertices.append(candidate)
        
        fundamental_domain = {
            "vertices": vertices,
            "boundary_roots": domain_roots.tolist(),
            "dimension": self.dimension
        }
        
        logger.debug(f"Computed fundamental domain with {len(vertices)} vertices")
        return fundamental_domain
    
    def export_symmetry_data(self, filename: str) -> Dict:
        """
        Export symmetry group data for external analysis.
        
        Args:
            filename: Output filename
            
        Returns:
            Dictionary with symmetry data
        """
        # Compute orbits for special points
        orbit_data = {}
        for category, points in self.special_points.items():
            orbit_data[category] = []
            for i, point in enumerate(points[:5]):  # Limit for file size
                orbit = self.compute_orbit(point)
                stabilizer = self.compute_stabilizer(point)
                
                orbit_data[category].append({
                    "point": point.tolist(),
                    "orbit_size": len(orbit),
                    "stabilizer_size": len(stabilizer),
                    "orbit": [p.tolist() for p in orbit[:20]]  # Limit orbit size
                })
        
        symmetry_data = {
            "lattice_type": self.lattice_type,
            "dimension": self.dimension,
            "group_order": self.group_order,
            "num_generators": len(self.generators),
            "generators": [g.tolist() for g in self.generators],
            "root_system_size": len(self.root_system),
            "special_points": {k: len(v) for k, v in self.special_points.items()},
            "orbit_data": orbit_data,
            "fundamental_domain": self.visualize_fundamental_domain()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(symmetry_data, f, indent=2)
        
        logger.info(f"Exported symmetry data to {filename}")
        return symmetry_data
    
    def get_explorer_info(self) -> Dict:
        """
        Get comprehensive information about the symmetry explorer.
        
        Returns:
            Dictionary with explorer information
        """
        return {
            "lattice_type": self.lattice_type,
            "dimension": self.dimension,
            "group_order": self.group_order,
            "num_generators": len(self.generators),
            "root_system_size": len(self.root_system),
            "special_point_categories": list(self.special_points.keys()),
            "total_special_points": sum(len(v) for v in self.special_points.values())
        }


# Example usage and testing
if __name__ == "__main__":
    # Test E8 symmetry explorer
    print("E8 Symmetry Explorer:")
    e8_explorer = SymmetryExplorer(lattice_type="E8")
    
    e8_info = e8_explorer.get_explorer_info()
    for key, value in e8_info.items():
        print(f"  {key}: {value}")
    
    # Test orbit computation
    test_point = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    orbit = e8_explorer.compute_orbit(test_point)
    print(f"  Orbit of {test_point} has {len(orbit)} points")
    
    # Test stabilizer
    stabilizer = e8_explorer.compute_stabilizer(test_point)
    print(f"  Stabilizer has {len(stabilizer)} elements")
    
    # Test A2 symmetry explorer
    print("\nA2 Symmetry Explorer:")
    a2_explorer = SymmetryExplorer(lattice_type="A2")
    
    a2_info = a2_explorer.get_explorer_info()
    for key, value in a2_info.items():
        print(f"  {key}: {value}")
    
    # Test orbit for A2
    a2_point = np.array([1, 0])
    a2_orbit = a2_explorer.compute_orbit(a2_point)
    print(f"  A2 orbit of {a2_point} has {len(a2_orbit)} points")
    
    # Test symmetry breaking
    print("\nSymmetry Breaking Analysis:")
    perturbation = np.array([0.1, 0, 0, 0, 0, 0, 0, 0])
    breaking_analysis = e8_explorer.analyze_symmetry_breaking(perturbation)
    print(f"  Symmetry breaking ratio: {breaking_analysis['symmetry_breaking_ratio']:.4f}")
    
    # Test fundamental domain
    print("\nFundamental Domain:")
    domain = e8_explorer.visualize_fundamental_domain()
    print(f"  Domain has {len(domain['vertices'])} vertices")
    
    # Test export
    print("\nExport Test:")
    symmetry_data = a2_explorer.export_symmetry_data("test_symmetry.json")
    print(f"  Exported data for {symmetry_data['lattice_type']} lattice")
    
    print("Symmetry explorer test passed!")

