"""
Mathematical utilities for lattice operations
"""

import numpy as np
from typing import Union, Tuple, List, Optional
from numba import jit, njit
import scipy.linalg
from scipy.special import gamma, zeta
import sympy as sp
from mpmath import mp

from ..core.exceptions import ComputationError


class MathUtils:
    """Mathematical utility functions for lattice operations"""
    
    @staticmethod
    @njit
    def gram_schmidt(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gram-Schmidt orthogonalization with Numba acceleration
        
        Args:
            vectors: Input vectors as rows of matrix
            
        Returns:
            Tuple of (orthogonal_vectors, coefficients)
        """
        n, m = vectors.shape
        orthogonal = np.zeros_like(vectors)
        coefficients = np.zeros((n, n))
        
        for i in range(n):
            orthogonal[i] = vectors[i].copy()
            for j in range(i):
                coeff = np.dot(vectors[i], orthogonal[j]) / np.dot(orthogonal[j], orthogonal[j])
                coefficients[i, j] = coeff
                orthogonal[i] -= coeff * orthogonal[j]
        
        return orthogonal, coefficients
    
    @staticmethod
    def lll_reduction(basis: np.ndarray, delta: float = 0.75) -> np.ndarray:
        """
        LLL lattice basis reduction algorithm
        
        Args:
            basis: Lattice basis vectors as rows
            delta: LLL parameter (typically 0.75)
            
        Returns:
            Reduced basis
        """
        try:
            # Use fpylll if available for better performance
            import fpylll
            from fpylll import IntegerMatrix, LLL
            
            # Convert to integer matrix
            int_basis = IntegerMatrix.from_matrix(basis.astype(int).tolist())
            
            # Perform LLL reduction
            LLL.reduction(int_basis, delta=delta)
            
            # Convert back to numpy array
            return np.array([[int_basis[i, j] for j in range(int_basis.ncols)] 
                           for i in range(int_basis.nrows)])
        
        except ImportError:
            # Fallback to basic implementation
            return MathUtils._basic_lll_reduction(basis, delta)
    
    @staticmethod
    def _basic_lll_reduction(basis: np.ndarray, delta: float = 0.75) -> np.ndarray:
        """Basic LLL reduction implementation"""
        n, m = basis.shape
        reduced_basis = basis.copy().astype(float)
        
        # Gram-Schmidt orthogonalization
        orthogonal, mu = MathUtils.gram_schmidt(reduced_basis)
        
        k = 1
        while k < n:
            # Size reduction
            for j in range(k-1, -1, -1):
                if abs(mu[k, j]) > 0.5:
                    reduced_basis[k] -= round(mu[k, j]) * reduced_basis[j]
                    # Recompute Gram-Schmidt
                    orthogonal, mu = MathUtils.gram_schmidt(reduced_basis)
            
            # Lovász condition
            if (np.dot(orthogonal[k], orthogonal[k]) >= 
                (delta - mu[k, k-1]**2) * np.dot(orthogonal[k-1], orthogonal[k-1])):
                k += 1
            else:
                # Swap vectors
                reduced_basis[[k, k-1]] = reduced_basis[[k-1, k]]
                orthogonal, mu = MathUtils.gram_schmidt(reduced_basis)
                k = max(k-1, 1)
        
        return reduced_basis
    
    @staticmethod
    @njit
    def babai_nearest_plane(target: np.ndarray, basis: np.ndarray, 
                           orthogonal: np.ndarray) -> np.ndarray:
        """
        Babai's nearest plane algorithm with O(n) complexity
        
        Args:
            target: Target vector
            basis: Lattice basis
            orthogonal: Gram-Schmidt orthogonalized basis
            
        Returns:
            Closest lattice point
        """
        n = basis.shape[0]
        coefficients = np.zeros(n)
        
        # Work backwards through the basis
        remainder = target.copy()
        for i in range(n-1, -1, -1):
            # Project onto orthogonal vector
            projection = np.dot(remainder, orthogonal[i]) / np.dot(orthogonal[i], orthogonal[i])
            coefficients[i] = round(projection)
            remainder -= coefficients[i] * basis[i]
        
        # Reconstruct lattice point
        lattice_point = np.zeros_like(target)
        for i in range(n):
            lattice_point += coefficients[i] * basis[i]
        
        return lattice_point
    
    @staticmethod
    def compute_theta_function(lattice_vectors: np.ndarray, tau: complex, 
                              max_terms: int = 1000) -> complex:
        """
        Compute theta function for lattice using modular forms
        
        Args:
            lattice_vectors: Lattice vectors
            tau: Complex parameter
            max_terms: Maximum number of terms in series
            
        Returns:
            Theta function value
        """
        if tau.imag <= 0:
            raise ComputationError("tau must have positive imaginary part")
        
        theta = 0.0 + 0.0j
        
        # Compute Gram matrix
        gram_matrix = np.dot(lattice_vectors, lattice_vectors.T)
        
        # Sum over lattice points (truncated)
        for n in range(-max_terms, max_terms + 1):
            if n == 0:
                continue
            
            # Compute quadratic form
            quadratic_form = 0.0
            for i in range(len(lattice_vectors)):
                for j in range(len(lattice_vectors)):
                    quadratic_form += n * gram_matrix[i, j] * n
            
            theta += np.exp(1j * np.pi * tau * quadratic_form)
        
        return theta
    
    @staticmethod
    def compute_packing_density(dimension: int, kissing_number: int, 
                               lattice_determinant: float) -> float:
        """
        Compute packing density of lattice
        
        Args:
            dimension: Lattice dimension
            kissing_number: Number of nearest neighbors
            lattice_determinant: Determinant of lattice
            
        Returns:
            Packing density
        """
        # Volume of unit sphere in n dimensions
        sphere_volume = (np.pi ** (dimension / 2)) / gamma(dimension / 2 + 1)
        
        # Packing density formula
        density = sphere_volume / (2 ** dimension * np.sqrt(lattice_determinant))
        
        return density
    
    @staticmethod
    def is_even_lattice(gram_matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if lattice is even (all diagonal entries are even integers)
        
        Args:
            gram_matrix: Gram matrix of lattice
            tolerance: Numerical tolerance
            
        Returns:
            True if lattice is even
        """
        diagonal = np.diag(gram_matrix)
        return np.all(np.abs(diagonal - 2 * np.round(diagonal / 2)) < tolerance)
    
    @staticmethod
    def is_unimodular(gram_matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if lattice is unimodular (determinant = ±1)
        
        Args:
            gram_matrix: Gram matrix of lattice
            tolerance: Numerical tolerance
            
        Returns:
            True if lattice is unimodular
        """
        det = np.linalg.det(gram_matrix)
        return abs(abs(det) - 1.0) < tolerance
    
    @staticmethod
    def compute_successive_minima(basis: np.ndarray, k: int = None) -> List[float]:
        """
        Compute successive minima of lattice
        
        Args:
            basis: Lattice basis
            k: Number of minima to compute (default: dimension)
            
        Returns:
            List of successive minima
        """
        if k is None:
            k = basis.shape[0]
        
        # This is a simplified implementation
        # In practice, would use more sophisticated algorithms
        gram_matrix = np.dot(basis, basis.T)
        eigenvalues = np.linalg.eigvals(gram_matrix)
        eigenvalues.sort()
        
        return [np.sqrt(max(0, ev)) for ev in eigenvalues[:k]]
    
    @staticmethod
    def modular_form_coefficient(n: int, weight: int, level: int = 1) -> complex:
        """
        Compute coefficient of modular form (simplified implementation)
        
        Args:
            n: Coefficient index
            weight: Weight of modular form
            level: Level of modular form
            
        Returns:
            Modular form coefficient
        """
        # This is a placeholder for more sophisticated modular form computations
        # In practice, would use specialized libraries like SageMath
        if weight == 4 and level == 1:  # Eisenstein series E_4
            if n == 0:
                return 1.0
            else:
                # Simplified coefficient formula
                return sum(d**3 for d in range(1, n+1) if n % d == 0)
        
        return 0.0 + 0.0j
    
    @staticmethod
    @njit
    def fast_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Fast matrix multiplication with Numba acceleration"""
        return np.dot(A, B)
    
    @staticmethod
    @njit
    def fast_vector_norm(v: np.ndarray) -> float:
        """Fast vector norm computation with Numba acceleration"""
        return np.sqrt(np.sum(v * v))
    
    @staticmethod
    def set_precision(precision: int):
        """Set arbitrary precision for mpmath computations"""
        mp.dps = precision

