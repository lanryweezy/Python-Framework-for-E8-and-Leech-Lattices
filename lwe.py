from typing import Optional, Tuple, List
import numpy as np
from e8leech.optimization.numba_optimizer import NumbaOptimizer
from e8leech.optimization.gpu_accelerator import GPUAccelerator

class LWECryptosystem:
    def __init__(self, dimension: int, modulus: int):
        self.dimension = dimension
        self.modulus = modulus
        self.numba_optimizer = NumbaOptimizer()
        self.gpu_accelerator = GPUAccelerator()

    def generate_keypair(self):
        # Generate a secret key s (a vector in Z_q^n)
        s = np.random.randint(0, self.modulus, self.dimension)

        # Generate a public matrix A (m x n matrix in Z_q)
        # For simplicity, let m = n for now
        A = np.random.randint(0, self.modulus, (self.dimension, self.dimension))

        # Generate an error vector e (a vector in Z_q^m with small coefficients)
        # For simplicity, using a small range for error
        e = np.random.randint(-5, 5, self.dimension)

        # Compute public key b = A * s + e (mod q)
        b = (A @ s + e) % self.modulus

        self.public_key = (A, b)
        self.private_key = s

        return self.public_key, self.private_key

    def encrypt(self, plaintext: bytes) -> List[Tuple[np.ndarray, int]]:
        if self.public_key is None:
            raise ValueError("Public key not generated. Call generate_keypair first.")

        A, b = self.public_key
        message_bits = self._bytes_to_bits(plaintext)
        ciphertext_parts = []

        # Encrypt bit by bit for simplicity
        for bit in message_bits:
            # Generate a random vector r (n-dimensional in Z_q)
            r = np.random.randint(0, self.modulus, self.dimension)

            # Generate a small error vector e1, e2
            # Increased error range slightly for more realistic simulation
            e1 = np.random.randint(-3, 3, self.dimension)
            e2 = np.random.randint(-3, 3)

            # Compute u = A^T * r + e1 (mod q)
            u = (A.T @ r + e1) % self.modulus

            # Compute v = b^T * r + e2 + bit * floor(q/2) (mod q)
            # For a clear signal, make the 'bit' contribution significant
            v = (b.T @ r + e2 + bit * (self.modulus // 2)) % self.modulus

            ciphertext_parts.append((u, v))

        return ciphertext_parts

    def decrypt(self, ciphertext: List[Tuple[np.ndarray, int]]) -> bytes:
        if self.private_key is None:
            raise ValueError("Private key not generated. Call generate_keypair first.")

        s = self.private_key
        decrypted_bits = []

        for u, v in ciphertext:
            # Compute decrypted_val = v - s^T * u (mod q)
            decrypted_val = (v - s.T @ u) % self.modulus

            # Normalize decrypted_val to be in [-modulus/2, modulus/2)
            # This step is crucial for correct interpretation of the signal
            decrypted_val = decrypted_val % self.modulus
            if decrypted_val > self.modulus / 2:
                decrypted_val -= self.modulus

            # Determine the original bit based on proximity to 0 or modulus/2
            # The original bit was added as bit * (self.modulus // 2)
            # So, if decrypted_val is close to 0, it was 0
            # If decrypted_val is close to modulus/2 (or -modulus/2), it was 1
            # The threshold should be exactly in the middle of 0 and modulus/2
            if abs(decrypted_val) < self.modulus / 4: # Closer to 0
                decrypted_bits.append(0)
            else: # Closer to modulus/2 (or -modulus/2)
                decrypted_bits.append(1)

        return self._bits_to_bytes(decrypted_bits)

    def _bytes_to_bits(self, data: bytes) -> List[int]:
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return bits

    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte |= (bits[i + j] << (7 - j))
            byte_array.append(byte)
        return bytes(byte_array)


class LWE:
    def __init__(self, dimension: int, modulus: int):
        self.lwe_cryptosystem = LWECryptosystem(dimension, modulus)
        self.lwe_cryptosystem.generate_keypair()

    def encrypt(self, plaintext: bytes) -> List[Tuple[np.ndarray, int]]:
        return self.lwe_cryptosystem.encrypt(plaintext)

    def decrypt(self, ciphertext: List[Tuple[np.ndarray, int]]) -> bytes:
        return self.lwe_cryptosystem.decrypt(ciphertext)


