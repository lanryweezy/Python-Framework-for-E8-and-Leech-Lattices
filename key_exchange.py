"""
Lattice-based Key Exchange implementation.

This module implements a quantum-resistant key exchange protocol based on lattice problems,
similar to the Ring-LWE key exchange or NewHope protocol.
"""

import numpy as np
import hashlib
from typing import Tuple, Optional
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class LatticeKeyExchange:
    """
    Lattice-based Key Exchange protocol.
    
    This implements a simplified version of a Ring-LWE based key exchange,
    which is quantum-resistant and provides forward secrecy.
    """
    
    def __init__(self, n: int = 1024, q: int = 12289, sigma: float = 3.2):
        """
        Initialize the lattice-based key exchange.
        
        Args:
            n: Dimension (must be power of 2 for Ring-LWE)
            q: Modulus (should be prime)
            sigma: Standard deviation for error distribution
        """
        self.n = n
        self.q = q
        self.sigma = sigma
        
        # Generate global polynomial a (shared parameter)
        self.a = self._generate_global_parameter()
        
        logger.info(f"Initialized lattice key exchange with n={n}, q={q}, sigma={sigma}")
    
    def _generate_global_parameter(self) -> np.ndarray:
        """
        Generate global parameter a for the key exchange.
        
        Returns:
            Global polynomial a
        """
        # In practice, this would be a standardized parameter
        np.random.seed(42)  # Fixed seed for reproducibility
        a = np.random.randint(0, self.q, size=self.n)
        np.random.seed()  # Reset seed
        return a
    
    def _polynomial_multiply_ntt(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Multiply two polynomials using Number Theoretic Transform (NTT).
        
        This is a simplified implementation. In practice, would use optimized NTT.
        
        Args:
            a, b: Coefficient vectors of polynomials
            
        Returns:
            Coefficient vector of product polynomial
        """
        # Simplified polynomial multiplication in Z[x]/(x^n + 1)
        result = np.zeros(self.n, dtype=int)
        
        for i in range(self.n):
            for j in range(self.n):
                coeff_idx = (i + j) % self.n
                sign = 1 if (i + j) < self.n else -1
                result[coeff_idx] = (result[coeff_idx] + sign * a[i] * b[j]) % self.q
        
        return result
    
    def _sample_error(self) -> np.ndarray:
        """
        Sample error polynomial from discrete Gaussian distribution.
        
        Returns:
            Error polynomial
        """
        # Sample from discrete Gaussian
        error = np.round(np.random.normal(0, self.sigma, self.n)).astype(int)
        return error % self.q
    
    def _sample_secret(self) -> np.ndarray:
        """
        Sample secret polynomial.
        
        Returns:
            Secret polynomial with small coefficients
        """
        # Sample from {-1, 0, 1} or small Gaussian
        secret = np.random.randint(-1, 2, size=self.n)
        return secret % self.q
    
    def _reconcile(self, v: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        Reconciliation function to extract shared key bits.
        
        Args:
            v: Value to reconcile
            signal: Reconciliation signal
            
        Returns:
            Reconciled bits
        """
        # Simplified reconciliation
        # In practice, would use more sophisticated error correction
        threshold = self.q // 4
        bits = np.zeros(self.n, dtype=int)
        
        for i in range(self.n):
            adjusted_v = (v[i] + signal[i] * threshold) % self.q
            bits[i] = 1 if adjusted_v > self.q // 2 else 0
        
        return bits
    
    def _generate_signal(self, v: np.ndarray) -> np.ndarray:
        """
        Generate reconciliation signal.
        
        Args:
            v: Value to generate signal for
            
        Returns:
            Reconciliation signal
        """
        # Simplified signal generation
        signal = np.zeros(self.n, dtype=int)
        threshold = self.q // 4
        
        for i in range(self.n):
            if v[i] < threshold or v[i] > 3 * threshold:
                signal[i] = 0
            else:
                signal[i] = 1
        
        return signal
    
    def generate_alice_message(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Alice's message for key exchange.
        
        Returns:
            Tuple (public_key, secret_key) where:
            - public_key: Alice's public value to send to Bob
            - secret_key: Alice's secret for later key derivation
        """
        # Alice samples secret and error
        s_a = self._sample_secret()
        e_a = self._sample_error()
        
        # Alice computes public value: p_a = a * s_a + e_a
        p_a = (self._polynomial_multiply_ntt(self.a, s_a) + e_a) % self.q
        
        logger.debug("Generated Alice's message")
        return p_a, s_a
    
    def generate_bob_message_and_key(self, alice_public: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bytes]:
        """
        Generate Bob's message and derive shared key.
        
        Args:
            alice_public: Alice's public value
            
        Returns:
            Tuple (bob_public, reconciliation_signal, shared_key) where:
            - bob_public: Bob's public value to send to Alice
            - reconciliation_signal: Signal to help Alice derive same key
            - shared_key: Bob's derived shared key
        """
        # Bob samples secret and errors
        s_b = self._sample_secret()
        e_b = self._sample_error()
        e_b_prime = self._sample_error()
        
        # Bob computes public value: p_b = a * s_b + e_b
        p_b = (self._polynomial_multiply_ntt(self.a, s_b) + e_b) % self.q
        
        # Bob computes shared value: v_b = alice_public * s_b + e_b'
        v_b = (self._polynomial_multiply_ntt(alice_public, s_b) + e_b_prime) % self.q
        
        # Bob generates reconciliation signal
        signal = self._generate_signal(v_b)
        
        # Bob derives key bits
        key_bits = self._reconcile(v_b, signal)
        
        # Hash key bits to get final shared key
        shared_key = hashlib.sha256(key_bits.tobytes()).digest()
        
        logger.debug("Generated Bob's message and derived shared key")
        return p_b, signal, shared_key
    
    def derive_alice_key(self, alice_secret: np.ndarray, bob_public: np.ndarray, 
                        reconciliation_signal: np.ndarray) -> bytes:
        """
        Derive Alice's shared key.
        
        Args:
            alice_secret: Alice's secret value
            bob_public: Bob's public value
            reconciliation_signal: Reconciliation signal from Bob
            
        Returns:
            Alice's derived shared key
        """
        # Alice computes shared value: v_a = bob_public * s_a
        v_a = self._polynomial_multiply_ntt(bob_public, alice_secret) % self.q
        
        # Alice uses reconciliation signal to derive key bits
        key_bits = self._reconcile(v_a, reconciliation_signal)
        
        # Hash key bits to get final shared key
        shared_key = hashlib.sha256(key_bits.tobytes()).digest()
        
        logger.debug("Alice derived shared key")
        return shared_key
    
    def full_key_exchange(self) -> Tuple[bytes, bytes]:
        """
        Perform a complete key exchange simulation.
        
        Returns:
            Tuple (alice_key, bob_key) - should be identical
        """
        # Step 1: Alice generates her message
        alice_public, alice_secret = self.generate_alice_message()
        
        # Step 2: Bob receives Alice's message and responds
        bob_public, reconciliation_signal, bob_key = self.generate_bob_message_and_key(alice_public)
        
        # Step 3: Alice derives the shared key
        alice_key = self.derive_alice_key(alice_secret, bob_public, reconciliation_signal)
        
        logger.info("Completed full key exchange")
        return alice_key, bob_key
    
    def get_security_level(self) -> int:
        """
        Estimate the security level in bits.
        
        Returns:
            Estimated security level
        """
        # Simplified security estimation based on lattice dimension
        return min(self.n // 8, 256)


# Example usage and testing
if __name__ == "__main__":
    # Create lattice key exchange
    kex = LatticeKeyExchange(n=512, q=12289, sigma=3.2)
    
    # Perform full key exchange
    alice_key, bob_key = kex.full_key_exchange()
    
    print(f"Alice's key: {alice_key.hex()[:32]}...")
    print(f"Bob's key:   {bob_key.hex()[:32]}...")
    print(f"Keys match: {alice_key == bob_key}")
    
    # Test multiple exchanges
    success_count = 0
    total_tests = 10
    
    for i in range(total_tests):
        alice_key, bob_key = kex.full_key_exchange()
        if alice_key == bob_key:
            success_count += 1
    
    print(f"Success rate: {success_count}/{total_tests} ({100*success_count/total_tests:.1f}%)")
    print(f"Security level: {kex.get_security_level()} bits")
    
    if success_count == total_tests:
        print("Lattice key exchange test passed!")
    else:
        print("Warning: Some key exchanges failed (this can happen with simplified reconciliation)")

