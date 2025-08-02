"""
Validation utilities for lattice operations
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from ..core.exceptions import ValidationError
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ValidationUtils:
    """Utilities for validating lattice properties and operations"""
    
    @staticmethod
    def validate_lattice_basis(basis: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Validate that basis vectors form a valid lattice basis
        
        Args:
            basis: Lattice basis vectors as rows
            tolerance: Numerical tolerance for validation
            
        Returns:
            True if valid basis
            
        Raises:
            ValidationError: If basis is invalid
        """
        if not isinstance(basis, np.ndarray):
            raise ValidationError("Basis must be a numpy array")
        
        if basis.ndim != 2:
            raise ValidationError("Basis must be a 2D array")
        
        n, m = basis.shape
        if n > m:
            raise ValidationError("Number of basis vectors cannot exceed dimension")
        
        # Check linear independence
        rank = np.linalg.matrix_rank(basis, tol=tolerance)
        if rank < n:
            raise ValidationError("Basis vectors are not linearly independent")
        
        # Check for zero vectors
        norms = np.linalg.norm(basis, axis=1)
        if np.any(norms < tolerance):
            raise ValidationError("Basis contains zero or near-zero vectors")
        
        logger.info(f"Validated lattice basis: {n} vectors in {m} dimensions")
        return True
    
    @staticmethod
    def validate_gram_matrix(gram_matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Validate Gram matrix properties
        
        Args:
            gram_matrix: Gram matrix to validate
            tolerance: Numerical tolerance
            
        Returns:
            True if valid Gram matrix
            
        Raises:
            ValidationError: If Gram matrix is invalid
        """
        if not isinstance(gram_matrix, np.ndarray):
            raise ValidationError("Gram matrix must be a numpy array")
        
        if gram_matrix.ndim != 2:
            raise ValidationError("Gram matrix must be 2D")
        
        n, m = gram_matrix.shape
        if n != m:
            raise ValidationError("Gram matrix must be square")
        
        # Check symmetry
        if not np.allclose(gram_matrix, gram_matrix.T, atol=tolerance):
            raise ValidationError("Gram matrix must be symmetric")
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvals(gram_matrix)
        if np.any(eigenvalues <= tolerance):
            raise ValidationError("Gram matrix must be positive definite")
        
        logger.info(f"Validated Gram matrix: {n}x{n}")
        return True
    
    @staticmethod
    def validate_even_lattice(gram_matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Validate that lattice is even (diagonal entries are even integers)
        
        Args:
            gram_matrix: Gram matrix of lattice
            tolerance: Numerical tolerance
            
        Returns:
            True if lattice is even
            
        Raises:
            ValidationError: If lattice is not even
        """
        ValidationUtils.validate_gram_matrix(gram_matrix, tolerance)
        
        diagonal = np.diag(gram_matrix)
        
        # Check that diagonal entries are integers
        if not np.allclose(diagonal, np.round(diagonal), atol=tolerance):
            raise ValidationError("Diagonal entries must be integers for even lattice")
        
        # Check that diagonal entries are even
        rounded_diagonal = np.round(diagonal).astype(int)
        if not np.all(rounded_diagonal % 2 == 0):
            raise ValidationError("Diagonal entries must be even integers")
        
        logger.info("Validated even lattice property")
        return True
    
    @staticmethod
    def validate_unimodular_lattice(gram_matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Validate that lattice is unimodular (determinant = ±1)
        
        Args:
            gram_matrix: Gram matrix of lattice
            tolerance: Numerical tolerance
            
        Returns:
            True if lattice is unimodular
            
        Raises:
            ValidationError: If lattice is not unimodular
        """
        ValidationUtils.validate_gram_matrix(gram_matrix, tolerance)
        
        det = np.linalg.det(gram_matrix)
        if abs(abs(det) - 1.0) > tolerance:
            raise ValidationError(f"Lattice determinant must be ±1, got {det}")
        
        logger.info(f"Validated unimodular lattice: determinant = {det}")
        return True
    
    @staticmethod
    def validate_root_system(roots: np.ndarray, expected_count: int, 
                           tolerance: float = 1e-10) -> bool:
        """
        Validate root system properties
        
        Args:
            roots: Root vectors
            expected_count: Expected number of roots
            tolerance: Numerical tolerance
            
        Returns:
            True if valid root system
            
        Raises:
            ValidationError: If root system is invalid
        """
        if not isinstance(roots, np.ndarray):
            raise ValidationError("Roots must be a numpy array")
        
        if roots.ndim != 2:
            raise ValidationError("Roots must be a 2D array")
        
        n_roots, dimension = roots.shape
        if n_roots != expected_count:
            raise ValidationError(f"Expected {expected_count} roots, got {n_roots}")
        
        # Check that roots have consistent norms (for simple root systems)
        norms = np.linalg.norm(roots, axis=1)
        if not np.allclose(norms, norms[0], atol=tolerance):
            logger.warning("Root norms are not consistent - may be valid for some root systems")
        
        # Check for duplicate roots
        for i in range(n_roots):
            for j in range(i + 1, n_roots):
                if np.allclose(roots[i], roots[j], atol=tolerance):
                    raise ValidationError(f"Duplicate roots found at indices {i} and {j}")
        
        logger.info(f"Validated root system: {n_roots} roots in {dimension} dimensions")
        return True
    
    @staticmethod
    def validate_theta_function_coefficients(coefficients: List[complex], 
                                           expected_properties: dict = None,
                                           tolerance: float = 1e-10) -> bool:
        """
        Validate theta function coefficients
        
        Args:
            coefficients: Theta function coefficients
            expected_properties: Expected properties (e.g., growth rate)
            tolerance: Numerical tolerance
            
        Returns:
            True if coefficients are valid
            
        Raises:
            ValidationError: If coefficients are invalid
        """
        if not isinstance(coefficients, (list, np.ndarray)):
            raise ValidationError("Coefficients must be a list or array")
        
        # Check for NaN or infinite values
        for i, coeff in enumerate(coefficients):
            if not np.isfinite(coeff):
                raise ValidationError(f"Invalid coefficient at index {i}: {coeff}")
        
        # Additional property checks if provided
        if expected_properties:
            if "first_coefficient" in expected_properties:
                expected_first = expected_properties["first_coefficient"]
                if abs(coefficients[0] - expected_first) > tolerance:
                    raise ValidationError(f"First coefficient should be {expected_first}, got {coefficients[0]}")
        
        logger.info(f"Validated theta function coefficients: {len(coefficients)} terms")
        return True
    
    @staticmethod
    def validate_packing_density(density: float, expected_range: Tuple[float, float] = None,
                               tolerance: float = 1e-6) -> bool:
        """
        Validate packing density value
        
        Args:
            density: Computed packing density
            expected_range: Expected range (min, max)
            tolerance: Numerical tolerance
            
        Returns:
            True if density is valid
            
        Raises:
            ValidationError: If density is invalid
        """
        if not isinstance(density, (int, float)):
            raise ValidationError("Packing density must be a number")
        
        if not (0 <= density <= 1):
            raise ValidationError(f"Packing density must be between 0 and 1, got {density}")
        
        if expected_range:
            min_expected, max_expected = expected_range
            if not (min_expected - tolerance <= density <= max_expected + tolerance):
                raise ValidationError(f"Packing density {density} not in expected range [{min_expected}, {max_expected}]")
        
        logger.info(f"Validated packing density: {density}")
        return True
    
    @staticmethod
    def validate_kissing_number(kissing_number: int, expected_value: int = None) -> bool:
        """
        Validate kissing number
        
        Args:
            kissing_number: Computed kissing number
            expected_value: Expected kissing number
            
        Returns:
            True if kissing number is valid
            
        Raises:
            ValidationError: If kissing number is invalid
        """
        if not isinstance(kissing_number, int):
            raise ValidationError("Kissing number must be an integer")
        
        if kissing_number <= 0:
            raise ValidationError("Kissing number must be positive")
        
        if expected_value is not None and kissing_number != expected_value:
            raise ValidationError(f"Expected kissing number {expected_value}, got {kissing_number}")
        
        logger.info(f"Validated kissing number: {kissing_number}")
        return True
    
    @staticmethod
    def validate_congruence_conditions(vectors: np.ndarray, modulus: int,
                                     condition_func: callable, tolerance: float = 1e-10) -> bool:
        """
        Validate congruence conditions for lattice vectors
        
        Args:
            vectors: Lattice vectors to check
            modulus: Modulus for congruence
            condition_func: Function that checks the congruence condition
            tolerance: Numerical tolerance
            
        Returns:
            True if all vectors satisfy the condition
            
        Raises:
            ValidationError: If congruence conditions are not satisfied
        """
        if not isinstance(vectors, np.ndarray):
            raise ValidationError("Vectors must be a numpy array")
        
        for i, vector in enumerate(vectors):
            if not condition_func(vector, modulus, tolerance):
                raise ValidationError(f"Vector at index {i} violates congruence condition")
        
        logger.info(f"Validated congruence conditions for {len(vectors)} vectors")
        return True

