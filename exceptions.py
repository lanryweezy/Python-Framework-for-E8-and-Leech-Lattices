"""
Exception classes for the E8 Leech Lattice Framework
"""

class LatticeError(Exception):
    """Base exception class for lattice-related errors"""
    pass

class ValidationError(LatticeError):
    """Raised when lattice validation fails"""
    pass

class ComputationError(LatticeError):
    """Raised when lattice computations fail"""
    pass

class ConfigurationError(LatticeError):
    """Raised when configuration is invalid"""
    pass

class OptimizationError(LatticeError):
    """Raised when optimization operations fail"""
    pass

class QuantumError(LatticeError):
    """Raised when quantum operations fail"""
    pass

class CryptoError(LatticeError):
    """Raised when cryptographic operations fail"""
    pass

