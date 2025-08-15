import numpy as np
from typing import Tuple, Optional

class E8Attention:
    """
    E8-based attention mechanism.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 use_e8_structure: bool = True):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_e8_structure = use_e8_structure

        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

        if use_e8_structure:
            self.e8_basis = self._get_e8_basis()
            self.e8_roots = self._get_e8_roots()

    def _get_e8_basis(self) -> np.ndarray:
        basis = np.zeros((8, 8))
        for i in range(7):
            basis[i, i] = 1.0
            basis[i, i+1] = -1.0
        basis[7, :] = 0.5
        return basis

    def _get_e8_roots(self) -> np.ndarray:
        roots = []
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        root = np.zeros(8)
                        root[i] = s1
                        root[j] = s2
                        roots.append(root)
        return np.array(roots[:64])

    def _e8_positional_encoding(self, seq_len: int) -> np.ndarray:
        pos_encoding = np.zeros((seq_len, self.d_model))
        if not self.use_e8_structure:
            for pos in range(seq_len):
                for i in range(0, self.d_model, 2):
                    pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                    if i + 1 < self.d_model:
                        pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / self.d_model)))
        else:
            for pos in range(seq_len):
                root_idx = pos % len(self.e8_roots)
                root = self.e8_roots[root_idx]
                for i in range(self.d_model):
                    dim_idx = i % 8
                    pos_encoding[pos, i] = root[dim_idx] * np.sin(pos / (10000 ** (i / self.d_model)))
        return pos_encoding

    def _scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray,
                                    V: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        scores = Q @ K.T / np.sqrt(self.d_head)
        if self.use_e8_structure:
            scores = self._apply_e8_structure(scores)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        attention_weights = self._softmax(scores)
        output = attention_weights @ V
        return output

    def _apply_e8_structure(self, scores: np.ndarray) -> np.ndarray:
        seq_len = scores.shape[0]
        structured_scores = scores.copy()
        for i in range(seq_len):
            for j in range(seq_len):
                pos_diff = abs(i - j)
                root_idx = pos_diff % len(self.e8_roots)
                root = self.e8_roots[root_idx]
                e8_factor = np.linalg.norm(root) / np.sqrt(2)
                structured_scores[i, j] *= (1 + 0.1 * e8_factor)
        return structured_scores

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        seq_len, d_model = x.shape
        pos_encoding = self._e8_positional_encoding(seq_len)
        x_with_pos = x + pos_encoding

        Q = x_with_pos @ self.W_q
        K = x_with_pos @ self.W_k
        V = x_with_pos @ self.W_v

        Q = Q.reshape(seq_len, self.n_heads, self.d_head)
        K = K.reshape(seq_len, self.n_heads, self.d_head)
        V = V.reshape(seq_len, self.n_heads, self.d_head)

        attention_outputs = []
        for head in range(self.n_heads):
            head_output = self._scaled_dot_product_attention(
                Q[:, head, :], K[:, head, :], V[:, head, :], mask
            )
            attention_outputs.append(head_output)

        concat_output = np.concatenate(attention_outputs, axis=-1)
        output = concat_output @ self.W_o
        return output
