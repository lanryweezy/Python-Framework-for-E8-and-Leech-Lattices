import numpy as np
import hashlib
from typing import Tuple, Optional

class LatticeKeyExchange:
    """
    Lattice-based Key Exchange protocol.
    """

    def __init__(self, n: int = 1024, q: int = 12289, sigma: float = 3.2):
        self.n = n
        self.q = q
        self.sigma = sigma
        self.a = self._generate_global_parameter()

    def _generate_global_parameter(self) -> np.ndarray:
        np.random.seed(42)
        a = np.random.randint(0, self.q, size=self.n)
        np.random.seed()
        return a

    def _polynomial_multiply_ntt(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                coeff_idx = (i + j) % self.n
                sign = 1 if (i + j) < self.n else -1
                result[coeff_idx] = (result[coeff_idx] + sign * a[i] * b[j]) % self.q
        return result

    def _sample_error(self) -> np.ndarray:
        return np.round(np.random.normal(0, self.sigma, self.n)).astype(int) % self.q

    def _sample_secret(self) -> np.ndarray:
        return np.random.randint(-1, 2, size=self.n) % self.q

    def _reconcile(self, v: np.ndarray, signal: np.ndarray) -> np.ndarray:
        threshold = self.q // 4
        bits = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            adjusted_v = (v[i] + signal[i] * threshold) % self.q
            bits[i] = 1 if adjusted_v > self.q // 2 else 0
        return bits

    def _generate_signal(self, v: np.ndarray) -> np.ndarray:
        signal = np.zeros(self.n, dtype=int)
        threshold = self.q // 4
        for i in range(self.n):
            if v[i] < threshold or v[i] > 3 * threshold:
                signal[i] = 0
            else:
                signal[i] = 1
        return signal

    def generate_alice_message(self) -> Tuple[np.ndarray, np.ndarray]:
        s_a = self._sample_secret()
        e_a = self._sample_error()
        p_a = (self._polynomial_multiply_ntt(self.a, s_a) + e_a) % self.q
        return p_a, s_a

    def generate_bob_message_and_key(self, alice_public: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bytes]:
        s_b = self._sample_secret()
        e_b = self._sample_error()
        e_b_prime = self._sample_error()
        p_b = (self._polynomial_multiply_ntt(self.a, s_b) + e_b) % self.q
        v_b = (self._polynomial_multiply_ntt(alice_public, s_b) + e_b_prime) % self.q
        signal = self._generate_signal(v_b)
        key_bits = self._reconcile(v_b, signal)
        shared_key = hashlib.sha256(key_bits.tobytes()).digest()
        return p_b, signal, shared_key

    def derive_alice_key(self, alice_secret: np.ndarray, bob_public: np.ndarray,
                        reconciliation_signal: np.ndarray) -> bytes:
        v_a = self._polynomial_multiply_ntt(bob_public, alice_secret) % self.q
        key_bits = self._reconcile(v_a, reconciliation_signal)
        shared_key = hashlib.sha256(key_bits.tobytes()).digest()
        return shared_key

    def full_key_exchange(self) -> Tuple[bytes, bytes]:
        alice_public, alice_secret = self.generate_alice_message()
        bob_public, reconciliation_signal, bob_key = self.generate_bob_message_and_key(alice_public)
        alice_key = self.derive_alice_key(alice_secret, bob_public, reconciliation_signal)
        return alice_key, bob_key
