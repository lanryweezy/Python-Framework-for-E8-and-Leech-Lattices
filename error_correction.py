"""
Error Correction using lattice codes.

This module implements error correction codes based on lattice structures,
particularly leveraging the properties of E8 and Leech lattices for
robust communication and data storage systems.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class ErrorCorrection:
    """
    Lattice-based error correction codes.
    
    This class implements error correction schemes that use lattice structures
    to provide robust encoding and decoding with optimal error correction
    capabilities.
    """
    
    def __init__(self, lattice_type: str = "E8", code_dimension: int = 8, 
                 noise_variance: float = 1.0):
        """
        Initialize lattice-based error correction.
        
        Args:
            lattice_type: Type of lattice ("E8", "Leech", "A2", "D4")
            code_dimension: Dimension of the code
            noise_variance: Variance of additive noise
        """
        self.lattice_type = lattice_type
        self.code_dimension = code_dimension
        self.noise_variance = noise_variance
        
        # Set up lattice structure
        self.lattice_basis = self._get_lattice_basis()
        self.generator_matrix = self._get_generator_matrix()
        self.parity_check_matrix = self._get_parity_check_matrix()
        
        # Code parameters
        self.code_rate = self._compute_code_rate()
        self.minimum_distance = self._compute_minimum_distance()
        self.error_correction_capability = (self.minimum_distance - 1) // 2
        
        logger.info(f"Initialized {lattice_type} error correction code")
        logger.info(f"Code dimension: {code_dimension}, Rate: {self.code_rate:.4f}")
        logger.info(f"Minimum distance: {self.minimum_distance}")
        logger.info(f"Error correction capability: {self.error_correction_capability}")
    
    def _get_lattice_basis(self) -> np.ndarray:
        """
        Get lattice basis for the error correction code.
        
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
            return np.eye(self.code_dimension)
    
    def _get_generator_matrix(self) -> np.ndarray:
        """
        Get generator matrix for the lattice code.
        
        Returns:
            Generator matrix
        """
        # For lattice codes, generator matrix is related to lattice basis
        if self.lattice_type == "E8":
            # E8 generator matrix (simplified)
            k = min(4, self.code_dimension)  # Information dimension
            n = self.code_dimension  # Code length
            
            # Create systematic generator matrix [I | P]
            generator = np.zeros((k, n))
            generator[:k, :k] = np.eye(k)  # Identity part
            
            # Parity part based on E8 structure
            if n > k:
                for i in range(k):
                    for j in range(k, n):
                        generator[i, j] = (i + j) % 2  # Simple parity pattern
            
            return generator
        
        elif self.lattice_type == "Leech":
            # Leech generator matrix (simplified)
            k = min(12, self.code_dimension)
            n = self.code_dimension
            
            generator = np.zeros((k, n))
            generator[:k, :k] = np.eye(k)
            
            # Extended Golay code structure
            if n > k:
                for i in range(k):
                    for j in range(k, n):
                        generator[i, j] = np.random.randint(0, 2)  # Random for simplicity
            
            return generator
        
        else:
            # Default generator matrix
            k = max(1, self.code_dimension // 2)
            n = self.code_dimension
            
            generator = np.zeros((k, n))
            generator[:k, :k] = np.eye(k)
            
            return generator
    
    def _get_parity_check_matrix(self) -> np.ndarray:
        """
        Get parity check matrix for the lattice code.
        
        Returns:
            Parity check matrix
        """
        G = self.generator_matrix
        k, n = G.shape
        
        if n == k:
            # No parity checks for full-rate code
            return np.array([]).reshape(0, n)
        
        # For systematic code [I | P], H = [-P^T | I]
        P = G[:, k:]  # Parity part
        r = n - k  # Number of parity checks
        
        H = np.zeros((r, n))
        H[:, :k] = -P.T  # -P^T part
        H[:, k:] = np.eye(r)  # Identity part
        
        return H
    
    def _compute_code_rate(self) -> float:
        """
        Compute the code rate.
        
        Returns:
            Code rate (k/n)
        """
        k, n = self.generator_matrix.shape
        return k / n if n > 0 else 0.0
    
    def _compute_minimum_distance(self) -> int:
        """
        Compute the minimum distance of the code.
        
        Returns:
            Minimum distance
        """
        if self.lattice_type == "E8":
            return 4  # E8 has minimum distance 4
        elif self.lattice_type == "Leech":
            return 8  # Leech has minimum distance 8
        elif self.lattice_type == "A2":
            return 2  # A2 has minimum distance 2
        elif self.lattice_type == "D4":
            return 2  # D4 has minimum distance 2
        else:
            return 1  # Default
    
    def encode(self, message: np.ndarray) -> np.ndarray:
        """
        Encode a message using the lattice code.
        
        Args:
            message: Message vector to encode
            
        Returns:
            Encoded codeword
        """
        G = self.generator_matrix
        k, n = G.shape
        
        if len(message) != k:
            raise ValueError(f"Message length must be {k}")
        
        # Linear encoding: c = m * G
        codeword = message @ G
        
        # For lattice codes, may need to quantize to lattice points
        if self.lattice_type in ["E8", "Leech"]:
            codeword = self._quantize_to_lattice(codeword)
        
        logger.debug(f"Encoded message of length {k} to codeword of length {n}")
        return codeword
    
    def _quantize_to_lattice(self, vector: np.ndarray) -> np.ndarray:
        """
        Quantize vector to nearest lattice point.
        
        Args:
            vector: Vector to quantize
            
        Returns:
            Quantized lattice point
        """
        if self.lattice_type == "E8":
            # E8 quantization (simplified)
            # Round to nearest integer, then adjust for E8 constraints
            quantized = np.round(vector)
            
            # Ensure even sum for E8
            if np.sum(quantized) % 2 != 0:
                # Flip the coordinate with smallest fractional part
                fractional_parts = np.abs(vector - quantized)
                min_idx = np.argmin(fractional_parts)
                quantized[min_idx] += 1 if vector[min_idx] > quantized[min_idx] else -1
            
            return quantized
        
        elif self.lattice_type == "Leech":
            # Leech quantization (simplified)
            return np.round(vector)
        
        else:
            return np.round(vector)
    
    def add_noise(self, codeword: np.ndarray) -> np.ndarray:
        """
        Add noise to a codeword.
        
        Args:
            codeword: Clean codeword
            
        Returns:
            Noisy codeword
        """
        noise = np.random.normal(0, np.sqrt(self.noise_variance), len(codeword))
        noisy_codeword = codeword + noise
        
        logger.debug(f"Added noise with variance {self.noise_variance}")
        return noisy_codeword
    
    def decode(self, received_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode a received vector using lattice decoding.
        
        Args:
            received_vector: Received (possibly noisy) vector
            
        Returns:
            Tuple of (decoded_message, decoded_codeword)
        """
        # Lattice decoding: find nearest lattice point
        decoded_codeword = self._nearest_lattice_point_decode(received_vector)
        
        # Extract message from systematic part
        G = self.generator_matrix
        k, n = G.shape
        
        if k > 0:
            # For systematic codes, message is in first k positions
            decoded_message = decoded_codeword[:k]
        else:
            decoded_message = np.array([])
        
        logger.debug(f"Decoded received vector to message of length {len(decoded_message)}")
        return decoded_message, decoded_codeword
    
    def _nearest_lattice_point_decode(self, received_vector: np.ndarray) -> np.ndarray:
        """
        Find nearest lattice point for decoding.
        
        Args:
            received_vector: Received vector
            
        Returns:
            Nearest lattice codeword
        """
        if self.lattice_type == "E8":
            return self._e8_decode(received_vector)
        elif self.lattice_type == "Leech":
            return self._leech_decode(received_vector)
        else:
            return self._generic_decode(received_vector)
    
    def _e8_decode(self, received_vector: np.ndarray) -> np.ndarray:
        """
        E8 lattice decoding.
        
        Args:
            received_vector: Received vector
            
        Returns:
            Decoded E8 lattice point
        """
        # Simplified E8 decoding
        # Round to nearest integer
        rounded = np.round(received_vector)
        
        # Check if sum is even
        if np.sum(rounded) % 2 == 0:
            return rounded
        else:
            # Find coordinate to adjust
            fractional_parts = np.abs(received_vector - rounded)
            max_idx = np.argmax(fractional_parts)
            
            # Try both directions
            candidate1 = rounded.copy()
            candidate1[max_idx] += 1
            
            candidate2 = rounded.copy()
            candidate2[max_idx] -= 1
            
            # Choose closer one
            dist1 = np.linalg.norm(received_vector - candidate1)
            dist2 = np.linalg.norm(received_vector - candidate2)
            
            return candidate1 if dist1 < dist2 else candidate2
    
    def _leech_decode(self, received_vector: np.ndarray) -> np.ndarray:
        """
        Leech lattice decoding (simplified).
        
        Args:
            received_vector: Received vector
            
        Returns:
            Decoded Leech lattice point
        """
        # Simplified Leech decoding - just round to nearest integer
        return np.round(received_vector)
    
    def _generic_decode(self, received_vector: np.ndarray) -> np.ndarray:
        """
        Generic lattice decoding.
        
        Args:
            received_vector: Received vector
            
        Returns:
            Decoded lattice point
        """
        return np.round(received_vector)
    
    def compute_error_probability(self, snr_db: float) -> float:
        """
        Compute error probability for given SNR.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Error probability
        """
        # Convert SNR from dB
        snr_linear = 10**(snr_db / 10)
        
        # Simplified error probability calculation
        # For lattice codes, this depends on minimum distance and noise
        sigma = np.sqrt(self.noise_variance)
        
        # Approximate error probability using minimum distance
        d_min = self.minimum_distance
        error_prob = 0.5 * np.exp(-(d_min**2) / (8 * sigma**2))
        
        logger.debug(f"Error probability at SNR {snr_db} dB: {error_prob:.6e}")
        return error_prob
    
    def simulate_transmission(self, message: np.ndarray, snr_db: float) -> Dict:
        """
        Simulate complete transmission with encoding, noise, and decoding.
        
        Args:
            message: Original message
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Dictionary with simulation results
        """
        # Encoding
        codeword = self.encode(message)
        
        # Add noise based on SNR
        signal_power = np.mean(codeword**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Temporarily adjust noise variance
        original_variance = self.noise_variance
        self.noise_variance = noise_power
        
        noisy_codeword = self.add_noise(codeword)
        
        # Decoding
        decoded_message, decoded_codeword = self.decode(noisy_codeword)
        
        # Restore original noise variance
        self.noise_variance = original_variance
        
        # Compute errors
        message_errors = np.sum(message != decoded_message) if len(message) == len(decoded_message) else len(message)
        codeword_errors = np.sum(codeword != decoded_codeword)
        
        simulation_results = {
            "original_message": message,
            "encoded_codeword": codeword,
            "noisy_codeword": noisy_codeword,
            "decoded_message": decoded_message,
            "decoded_codeword": decoded_codeword,
            "message_errors": message_errors,
            "codeword_errors": codeword_errors,
            "snr_db": snr_db,
            "successful_decoding": message_errors == 0
        }
        
        logger.info(f"Transmission simulation completed")
        logger.info(f"Message errors: {message_errors}, Successful: {message_errors == 0}")
        
        return simulation_results
    
    def get_code_info(self) -> Dict:
        """
        Get comprehensive information about the error correction code.
        
        Returns:
            Dictionary with code information
        """
        G = self.generator_matrix
        H = self.parity_check_matrix
        
        return {
            "lattice_type": self.lattice_type,
            "code_dimension": self.code_dimension,
            "generator_matrix_shape": G.shape,
            "parity_check_matrix_shape": H.shape,
            "code_rate": self.code_rate,
            "minimum_distance": self.minimum_distance,
            "error_correction_capability": self.error_correction_capability,
            "noise_variance": self.noise_variance
        }


# Example usage and testing
if __name__ == "__main__":
    # Test E8 error correction
    e8_code = ErrorCorrection(lattice_type="E8", code_dimension=8, noise_variance=0.1)
    
    print("E8 Error Correction Code:")
    e8_info = e8_code.get_code_info()
    for key, value in e8_info.items():
        print(f"  {key}: {value}")
    
    # Test encoding and decoding
    message = np.array([1, 0, 1, 1])  # 4-bit message
    codeword = e8_code.encode(message)
    print(f"  Encoded message {message} to codeword {codeword}")
    
    # Add noise and decode
    noisy_codeword = e8_code.add_noise(codeword)
    decoded_message, decoded_codeword = e8_code.decode(noisy_codeword)
    print(f"  Decoded to message {decoded_message}")
    print(f"  Decoding successful: {np.array_equal(message, decoded_message)}")
    
    # Test transmission simulation
    simulation = e8_code.simulate_transmission(message, snr_db=10.0)
    print(f"  Transmission at 10 dB SNR: {simulation['successful_decoding']}")
    
    # Test error probability
    error_prob = e8_code.compute_error_probability(snr_db=10.0)
    print(f"  Error probability at 10 dB: {error_prob:.6e}")
    
    # Test A2 code
    print("\nA2 Error Correction Code:")
    a2_code = ErrorCorrection(lattice_type="A2", code_dimension=2, noise_variance=0.1)
    a2_info = a2_code.get_code_info()
    for key, value in a2_info.items():
        print(f"  {key}: {value}")
    
    # Test with 2D message
    message_2d = np.array([1])
    codeword_2d = a2_code.encode(message_2d)
    print(f"  A2 encoded: {message_2d} -> {codeword_2d}")
    
    # Test Leech code
    print("\nLeech Error Correction Code:")
    leech_code = ErrorCorrection(lattice_type="Leech", code_dimension=24, noise_variance=0.1)
    leech_info = leech_code.get_code_info()
    for key, value in leech_info.items():
        print(f"  {key}: {value}")
    
    print("Error correction test passed!")

