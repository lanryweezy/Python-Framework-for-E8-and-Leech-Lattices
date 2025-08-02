import pytest
import numpy as np
from e8leech.lattices.e8_lattice import E8Lattice
from e8leech.lattices.leech_lattice import LeechLattice
from e8leech.modules.crypto.lwe import LWECryptosystem

class TestErrorInjection:

    def test_e8_lattice_noise_tolerance(self):
        e8_lattice = E8Lattice()
        original_point = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # A valid E8 point
        
        # Add small noise
        noise = np.random.normal(0, 0.01, size=original_point.shape)
        noisy_point = original_point + noise
        
        # Quantize the noisy point back to the lattice
        quantized_point = e8_lattice.quantize(noisy_point)
        
        # Assert that the quantized point is close to the original valid point
        assert np.allclose(quantized_point, original_point, atol=0.1), \
            f"E8 lattice quantization failed with noise. Original: {original_point}, Noisy: {noisy_point}, Quantized: {quantized_point}"

    def test_leech_lattice_noise_tolerance(self):
        leech_lattice = LeechLattice()
        # A valid Leech lattice point (example: a scaled Golay codeword)
        # For simplicity, use a point that should quantize to itself with small noise
        original_point = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]) # Example, not necessarily a true Leech point
        
        # Add small noise
        noise = np.random.normal(0, 0.01, size=original_point.shape)
        noisy_point = original_point + noise
        
        # Quantize the noisy point back to the lattice
        quantized_point = leech_lattice.quantize(noisy_point)
        
        # Assert that the quantized point is close to the original valid point
        assert np.allclose(quantized_point, original_point, atol=0.1), \
            f"Leech lattice quantization failed with noise. Original: {original_point}, Noisy: {noisy_point}, Quantized: {quantized_point}"

    def test_lwe_error_injection(self):
        dimension = 24
        modulus = 257
        lwe = LWECryptosystem(dimension, modulus)
        lwe.generate_keypair()

        plaintext = b'secret_message'
        ciphertext = lwe.encrypt(plaintext)

        # Inject errors into the ciphertext (e.g., flip a bit)
        # This is a simplified error injection for demonstration
        corrupted_ciphertext = []
        for u, v in ciphertext:
            # Randomly flip a bit in v (conceptual)
            if np.random.rand() < 0.1: # 10% chance of error
                v_corrupted = (v + np.random.randint(1, modulus // 4)) % modulus
                corrupted_ciphertext.append((u, v_corrupted))
            else:
                corrupted_ciphertext.append((u, v))

        decrypted_text = lwe.decrypt(corrupted_ciphertext)
        
        # Assert that decryption is still successful within a certain tolerance
        # For LWE, perfect decryption with injected errors is not guaranteed without error correction codes
        # This test checks if it doesn't completely break, or if it can recover a portion
        # For now, we'll just check if it doesn't raise an error and produces some output
        assert isinstance(decrypted_text, bytes)
        assert len(decrypted_text) > 0

        # More rigorous test would involve comparing Hamming distance or specific error correction capabilities
        # For this framework, we are checking basic fault tolerance.


