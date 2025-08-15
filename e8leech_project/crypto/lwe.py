from typing import Optional, Tuple, List
import numpy as np

class LWECryptosystem:
    def __init__(self, dimension: int, modulus: int):
        self.dimension = dimension
        self.modulus = modulus
        self.public_key = None
        self.private_key = None

    def generate_keypair(self):
        s = np.random.randint(0, self.modulus, self.dimension)
        A = np.random.randint(0, self.modulus, (self.dimension, self.dimension))
        e = np.random.randint(-5, 5, self.dimension)
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

        for bit in message_bits:
            r = np.random.randint(0, self.modulus, self.dimension)
            e1 = np.random.randint(-3, 3, self.dimension)
            e2 = np.random.randint(-3, 3)
            u = (A.T @ r + e1) % self.modulus
            v = (b.T @ r + e2 + bit * (self.modulus // 2)) % self.modulus
            ciphertext_parts.append((u, v))

        return ciphertext_parts

    def decrypt(self, ciphertext: List[Tuple[np.ndarray, int]]) -> bytes:
        if self.private_key is None:
            raise ValueError("No private key available. Generate keypair first.")

        s = self.private_key
        q = self.modulus
        half_q = q // 2
        quarter_q = q // 4
        decrypted_bits = []

        for u, v in ciphertext:
            decrypted_val = int((v - int(s.T @ u)) % q)
            if decrypted_val > half_q:
                decrypted_val -= q
            if abs(decrypted_val) <= quarter_q:
                decrypted_bits.append(0)
            else:
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
