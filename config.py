
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import json
import os
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass
class OptimizationConfig:
    """Configuration for optimization settings"""
    use_gpu: bool = True
    use_numba: bool = True
    use_ray: bool = True
    use_fp16: bool = False
    batch_size: int = 1000
    cache_size: int = 10000
    parallel_workers: int = -1  # -1 means use all available cores


@dataclass
class QuantumConfig:
    """Configuration for quantum computing settings"""
    backend: str = "qiskit"  # qiskit, cirq, pennylane
    simulator: str = "aer_simulator"
    shots: int = 1024
    optimization_level: int = 1


@dataclass
class CryptoConfig:
    """Configuration for cryptographic settings"""
    security_level: int = 128  # bits
    use_quantum_resistant: bool = True
    key_size: int = 2048


@dataclass
class ValidationConfig:
    """Configuration for validation settings"""
    tolerance: float = 1e-10
    check_unimodularity: bool = True
    check_even_lattice: bool = True
    validate_theta_functions: bool = True


@dataclass
class LatticeSpecificConfig:
    """Configuration for lattice-specific settings"""
    generate_full_leech_roots: bool = False


@dataclass
class LatticeConfig:
    """Main configuration class for lattice operations"""
    
    # Basic settings
    precision: str = "float64"  # float16, float32, float64
    random_seed: Optional[int] = None
    
    # Optimization settings
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Quantum settings
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    
    # Crypto settings
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    
    # Validation settings
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Lattice-specific settings
    lattice_specific: LatticeSpecificConfig = field(default_factory=LatticeSpecificConfig)
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Storage settings
    cache_dir: str = "~/.e8leech_cache"
    data_dir: str = "~/.e8leech_data"
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Expand user paths
        self.cache_dir = os.path.expanduser(self.cache_dir)
        self.data_dir = os.path.expanduser(self.data_dir)
        
        # Create directories if they don't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate precision
        if self.precision not in ["float16", "float32", "float64"]:
            raise ConfigurationError(f"Invalid precision: {self.precision}")
        
        # Validate optimization settings
        if self.optimization.parallel_workers == -1:
            self.optimization.parallel_workers = os.cpu_count()
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "LatticeConfig":
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LatticeConfig":
        """Create configuration from dictionary"""
        try:
            # Handle nested configurations
            if "optimization" in config_dict:
                config_dict["optimization"] = OptimizationConfig(**config_dict["optimization"])
            if "quantum" in config_dict:
                config_dict["quantum"] = QuantumConfig(**config_dict["quantum"])
            if "crypto" in config_dict:
                config_dict["crypto"] = CryptoConfig(**config_dict["crypto"])
            if "validation" in config_dict:
                config_dict["validation"] = ValidationConfig(**config_dict["validation"])
            if "lattice_specific" in config_dict:
                config_dict["lattice_specific"] = LatticeSpecificConfig(**config_dict["lattice_specific"])
            
            return cls(**config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration from dict: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result
    
    def update(self, **kwargs) -> "LatticeConfig":
        """Update configuration with new values"""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)


# Default configuration instance
DEFAULT_CONFIG = LatticeConfig()



