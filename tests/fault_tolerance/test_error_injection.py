import pytest
import numpy as np
from e8leech_project.lattices.e8_lattice import E8Lattice
from e8leech_project.lattices.leech_lattice import LeechLattice
from e8leech_project.crypto.lwe import LWECryptosystem

class TestErrorInjection:

    def test_e8_lattice_noise_tolerance(self):
        e8_lattice = E8Lattice()
        original_point = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        noise = np.random.normal(0, 0.01, size=original_point.shape)
        noisy_point = original_point + noise
        quantized_point = e8_lattice.quantize(np.array([noisy_point]))[0]
        assert np.allclose(quantized_point, original_point, atol=0.1)

    def test_leech_lattice_noise_tolerance(self):
        leech_lattice = LeechLattice()
        original_point = np.array([2.0] * 24)
        noise = np.random.normal(0, 0.01, size=original_point.shape)
        noisy_point = original_point + noise
        quantized_point = leech_lattice.quantize(np.array([noisy_point]))[0]
        assert np.allclose(quantized_point, original_point, atol=0.1)

    def test_lwe_error_injection(self):
        lwe = LWECryptosystem(24, 257)
        lwe.generate_keypair()
        plaintext = b'secret_message'
        ciphertext = lwe.encrypt(plaintext)
        corrupted_ciphertext = []
        for u, v in ciphertext:
            if np.random.rand() < 0.1:
                v_corrupted = (v + np.random.randint(1, 257 // 4)) % 257
                corrupted_ciphertext.append((u, v_corrupted))
            else:
                corrupted_ciphertext.append((u, v))
        decrypted_text = lwe.decrypt(corrupted_ciphertext)
        assert isinstance(decrypted_text, bytes)
        assert len(decrypted_text) > 0
