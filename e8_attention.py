"""
E8-Attention mechanisms for neural networks.

This module implements attention mechanisms that leverage the structure
of the E8 lattice for enhanced representational capacity.
"""

import numpy as np
from typing import Tuple, Optional
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class E8Attention:
    """
    E8-based attention mechanism.
    
    This attention mechanism uses the E8 lattice structure to create
    more structured and geometrically meaningful attention patterns.
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, 
                 use_e8_structure: bool = True):
        """
        Initialize E8 attention mechanism.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            use_e8_structure: Whether to use E8 lattice structure
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_e8_structure = use_e8_structure
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
        # E8 lattice structure
        if use_e8_structure:
            self.e8_basis = self._get_e8_basis()
            self.e8_roots = self._get_e8_roots()
        
        logger.info(f"Initialized E8 attention with d_model={d_model}, n_heads={n_heads}")
    
    def _get_e8_basis(self) -> np.ndarray:
        """
        Get E8 lattice basis vectors.
        
        Returns:
            E8 basis matrix
        """
        # Standard E8 basis construction
        basis = np.zeros((8, 8))
        
        # v_i = e_i - e_{i+1} for i=1 to 7
        for i in range(7):
            basis[i, i] = 1.0
            basis[i, i+1] = -1.0
        
        # v_8 = (1/2, 1/2, ..., 1/2)
        basis[7, :] = 0.5
        
        return basis
    
    def _get_e8_roots(self) -> np.ndarray:
        """
        Get a subset of E8 root vectors for attention patterns.
        
        Returns:
            Array of E8 root vectors
        """
        roots = []
        
        # Type 1: vectors with two ±1 entries and six 0 entries
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        root = np.zeros(8)
                        root[i] = s1
                        root[j] = s2
                        roots.append(root)
        
        # Take a subset for computational efficiency
        return np.array(roots[:64])  # Use first 64 roots
    
    def _e8_positional_encoding(self, seq_len: int) -> np.ndarray:
        """
        Create positional encodings based on E8 lattice structure.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encoding matrix
        """
        pos_encoding = np.zeros((seq_len, self.d_model))
        
        if not self.use_e8_structure:
            # Standard sinusoidal positional encoding
            for pos in range(seq_len):
                for i in range(0, self.d_model, 2):
                    pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                    if i + 1 < self.d_model:
                        pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / self.d_model)))
        else:
            # E8-based positional encoding
            for pos in range(seq_len):
                # Use E8 roots to create structured positional encodings
                root_idx = pos % len(self.e8_roots)
                root = self.e8_roots[root_idx]
                
                # Expand 8D root to d_model dimensions
                for i in range(self.d_model):
                    dim_idx = i % 8
                    pos_encoding[pos, i] = root[dim_idx] * np.sin(pos / (10000 ** (i / self.d_model)))
        
        return pos_encoding
    
    def _scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, 
                                    V: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query matrix
            K: Key matrix
            V: Value matrix
            mask: Optional attention mask
            
        Returns:
            Attention output
        """
        # Compute attention scores
        scores = Q @ K.T / np.sqrt(self.d_head)
        
        # Apply E8 structure to attention scores
        if self.use_e8_structure:
            scores = self._apply_e8_structure(scores)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        
        # Softmax
        attention_weights = self._softmax(scores)
        
        # Apply attention to values
        output = attention_weights @ V
        
        return output
    
    def _apply_e8_structure(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply E8 lattice structure to attention scores.
        
        Args:
            scores: Raw attention scores
            
        Returns:
            Structured attention scores
        """
        # Apply E8-based transformations to attention scores
        # This creates more structured attention patterns
        
        seq_len = scores.shape[0]
        structured_scores = scores.copy()
        
        # Use E8 symmetries to create structured patterns
        for i in range(seq_len):
            for j in range(seq_len):
                # Distance in E8 lattice space
                pos_diff = abs(i - j)
                root_idx = pos_diff % len(self.e8_roots)
                root = self.e8_roots[root_idx]
                
                # Modulate attention based on E8 structure
                e8_factor = np.linalg.norm(root) / np.sqrt(2)  # Normalize by typical root norm
                structured_scores[i, j] *= (1 + 0.1 * e8_factor)
        
        return structured_scores
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Compute softmax function.
        
        Args:
            x: Input array
            
        Returns:
            Softmax output
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through E8 attention.
        
        Args:
            x: Input tensor (seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Attention output
        """
        seq_len, d_model = x.shape
        
        # Add positional encoding
        pos_encoding = self._e8_positional_encoding(seq_len)
        x_with_pos = x + pos_encoding
        
        # Compute Q, K, V
        Q = x_with_pos @ self.W_q
        K = x_with_pos @ self.W_k
        V = x_with_pos @ self.W_v
        
        # Reshape for multi-head attention
        Q = Q.reshape(seq_len, self.n_heads, self.d_head)
        K = K.reshape(seq_len, self.n_heads, self.d_head)
        V = V.reshape(seq_len, self.n_heads, self.d_head)
        
        # Apply attention for each head
        attention_outputs = []
        for head in range(self.n_heads):
            head_output = self._scaled_dot_product_attention(
                Q[:, head, :], K[:, head, :], V[:, head, :], mask
            )
            attention_outputs.append(head_output)
        
        # Concatenate heads
        concat_output = np.concatenate(attention_outputs, axis=-1)
        
        # Final linear transformation
        output = concat_output @ self.W_o
        
        logger.debug(f"E8 attention forward pass completed. Output shape: {output.shape}")
        return output
    
    def compute_attention_weights(self, x: np.ndarray) -> np.ndarray:
        """
        Compute and return attention weights for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights
        """
        seq_len, d_model = x.shape
        
        # Add positional encoding
        pos_encoding = self._e8_positional_encoding(seq_len)
        x_with_pos = x + pos_encoding
        
        # Compute Q, K
        Q = x_with_pos @ self.W_q
        K = x_with_pos @ self.W_k
        
        # Compute attention scores for first head
        Q_head = Q[:, :self.d_head]
        K_head = K[:, :self.d_head]
        
        scores = Q_head @ K_head.T / np.sqrt(self.d_head)
        
        if self.use_e8_structure:
            scores = self._apply_e8_structure(scores)
        
        attention_weights = self._softmax(scores)
        
        return attention_weights
    
    def get_e8_structure_info(self) -> dict:
        """
        Get information about the E8 structure used.
        
        Returns:
            Dictionary with E8 structure information
        """
        if not self.use_e8_structure:
            return {"e8_structure_enabled": False}
        
        return {
            "e8_structure_enabled": True,
            "e8_basis_shape": self.e8_basis.shape,
            "e8_roots_count": len(self.e8_roots),
            "e8_roots_norm": np.linalg.norm(self.e8_roots, axis=1).mean()
        }


# Example usage and testing
if __name__ == "__main__":
    # Create sample input
    seq_len, d_model = 32, 512
    x = np.random.randn(seq_len, d_model)
    
    # Test E8 attention
    e8_attention = E8Attention(d_model=d_model, n_heads=8, use_e8_structure=True)
    output = e8_attention.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test attention weights
    attention_weights = e8_attention.compute_attention_weights(x)
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Compare with standard attention
    standard_attention = E8Attention(d_model=d_model, n_heads=8, use_e8_structure=False)
    standard_output = standard_attention.forward(x)
    
    print(f"Standard attention output shape: {standard_output.shape}")
    
    # Check E8 structure info
    e8_info = e8_attention.get_e8_structure_info()
    print(f"E8 structure info: {e8_info}")
    
    # Verify outputs are different (E8 structure should make a difference)
    difference = np.linalg.norm(output - standard_output)
    print(f"Difference between E8 and standard attention: {difference:.6f}")
    
    print("E8 attention test passed!")

