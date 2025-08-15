import numpy as np
import hashlib
from typing import Tuple, Optional

class BLISSSignature:
    """
    BLISS (Bimodal Lattice Signature Scheme) implementation.

    WARNING: This implementation is for educational purposes only and is not
    secure for production use. It lacks critical security features,
    including proper verification and secure sampling, which makes it
    vulnerable to attacks.
    """

    def __init__(self, n: int = 512, q: int = 12289, sigma: float = 215.0, kappa: int = 23):
        self.n = n
        self.q = q
        self.sigma = sigma
        self.kappa = kappa
        self.private_key = None
        self.public_key = None

    def _polynomial_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                coeff_idx = (i + j) % self.n
                sign = 1 if (i + j) < self.n else -1
                result[coeff_idx] = (result[coeff_idx] + sign * a[i] * b[j]) % self.q
        return result

    def _gaussian_sample(self, size: int) -> np.ndarray:
        return np.round(np.random.normal(0, self.sigma, size)).astype(int) % self.q

    def _uniform_sample(self, size: int) -> np.ndarray:
        return np.random.randint(0, self.q, size=size)

    def generate_keypair(self) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        s1 = self._gaussian_sample(self.n)
        s2 = self._gaussian_sample(self.n)
        a = self._uniform_sample(self.n)
        a_q = (2 * self._polynomial_multiply(a, s1) + s2) % self.q

        self.private_key = (s1, s2)
        self.public_key = a_q

        return (s1, s2), a_q

    def _hash_to_challenge(self, message: bytes, commitment: np.ndarray) -> np.ndarray:
        hash_input = message + commitment.tobytes()
        hash_digest = hashlib.sha256(hash_input).digest()

        challenge = np.zeros(self.n, dtype=int)
        for i in range(min(self.n, len(hash_digest))):
            challenge[i] = hash_digest[i] % 3 - 1

        return challenge

    def sign(self, message: bytes,
             private_key: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        if private_key is None:
            if self.private_key is None:
                raise ValueError("No private key available. Generate keypair first.")
            s1, s2 = self.private_key
        else:
            s1, s2 = private_key

        max_attempts = 1000
        for attempt in range(max_attempts):
            y1 = self._gaussian_sample(self.n)
            y2 = self._gaussian_sample(self.n)
            commitment = (y1 + y2) % self.q
            c = self._hash_to_challenge(message, commitment)
            z1 = (y1 + self._polynomial_multiply(c, s1)) % self.q
            z2 = (y2 + self._polynomial_multiply(c, s2)) % self.q

            if np.linalg.norm(z1) < self.sigma * np.sqrt(self.n) and \
               np.linalg.norm(z2) < self.sigma * np.sqrt(self.n):
                return z1, z2

        raise RuntimeError("Failed to generate signature after maximum attempts")

    def verify(self, message: bytes, signature: Tuple[np.ndarray, np.ndarray],
               public_key: Optional[np.ndarray] = None) -> bool:
        if public_key is None:
            if self.public_key is None:
                raise ValueError("No public key available. Generate keypair first.")
            a_q = self.public_key
        else:
            a_q = public_key

        z1, z2 = signature
        commitment = (z1 + z2) % self.q
        c = self._hash_to_challenge(message, commitment)

        if np.linalg.norm(z1) >= self.sigma * np.sqrt(self.n) * 2 or \
           np.linalg.norm(z2) >= self.sigma * np.sqrt(self.n) * 2:
            return False

        return True
