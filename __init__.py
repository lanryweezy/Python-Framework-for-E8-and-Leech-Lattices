"""
Core module for E8 Leech Lattice Framework

Contains base classes and fundamental lattice operations.
"""

from .base_lattice import BaseLattice
from .config import LatticeConfig
from .exceptions import LatticeError, ValidationError, ComputationError

__all__ = [
    "BaseLattice",
    "LatticeConfig", 
    "LatticeError",
    "ValidationError",
    "ComputationError"
]

