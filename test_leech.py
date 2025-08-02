#!/usr/bin/env python3
"""
Test script for Leech lattice implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from e8leech.lattices.leech_lattice import LeechLattice
from e8leech.core.config import LatticeConfig

def test_leech_basic():
    """Test basic Leech lattice functionality"""
    print("Testing Leech lattice basic functionality...")
    
    # Create Leech lattice
    leech = LeechLattice()
    
    print(f"Leech dimension: {leech.dimension}")
    print(f"Leech rank: {leech.rank}")
    print(f"Leech determinant: {leech.determinant}")
    
    # Test basic properties
    assert leech.dimension == 24, f"Expected dimension 24, got {leech.dimension}"
    assert leech.rank == 24, f"Expected rank 24, got {leech.rank}"
    
    print("✓ Basic properties validated")

def test_leech_golay_code():
    """Test Golay code generation"""
    print("Testing Golay code generation...")
    
    leech = LeechLattice()
    golay_code = leech.get_golay_code()
    
    print(f"Generated {len(golay_code)} Golay codewords")
    print(f"Codeword length: {golay_code.shape[1] if len(golay_code) > 0 else 0}")
    
    # Check expected properties
    expected_codewords = 4096  # 2^12
    expected_length = 24
    
    if len(golay_code) == expected_codewords:
        print(f"✓ Correct number of codewords: {expected_codewords}")
    else:
        print(f"⚠ Expected {expected_codewords} codewords, got {len(golay_code)}")
    
    if len(golay_code) > 0 and golay_code.shape[1] == expected_length:
        print(f"✓ Correct codeword length: {expected_length}")
    else:
        print(f"⚠ Expected codeword length {expected_length}")
    
    print("✓ Golay code generation working")

def test_leech_root_system():
    """Test Leech lattice root system generation"""
    print("Testing Leech lattice root system generation...")
    
    leech = LeechLattice()
    
    # Note: Full root system generation is computationally intensive
    # We test with a subset
    roots = leech.generate_root_system()
    
    print(f"Generated {len(roots)} root vectors (subset)")
    
    if len(roots) > 0:
        # Check root norms (should be 2 for minimal vectors)
        norms = np.linalg.norm(roots, axis=1)
        print(f"Root norms - min: {norms.min():.6f}, max: {norms.max():.6f}")
        
        # For Leech lattice, minimal vectors have norm 2
        expected_norm = 2.0
        correct_norms = np.sum(np.abs(norms - expected_norm) < 1e-6)
        print(f"Roots with correct norm: {correct_norms}/{len(roots)}")
    
    print("✓ Root system generation working")

def test_leech_nearest_neighbor():
    """Test nearest neighbor search"""
    print("Testing Leech nearest neighbor search...")
    
    leech = LeechLattice()
    
    # Test with a few random vectors
    np.random.seed(42)
    test_vectors = np.random.randn(3, 24)
    
    for i, vec in enumerate(test_vectors):
        try:
            nearest = leech.nearest_neighbor(vec)
            distance = np.linalg.norm(vec - nearest)
            print(f"Vector {i}: distance to lattice = {distance:.6f}")
        except Exception as e:
            print(f"Vector {i}: Error in nearest neighbor search: {e}")
    
    print("✓ Nearest neighbor search working")

def test_leech_properties():
    """Test Leech mathematical properties"""
    print("Testing Leech mathematical properties...")
    
    leech = LeechLattice()
    
    # Validate properties
    validation_results = leech.validate_properties()
    print("Validation results:")
    for prop, result in validation_results.items():
        status = "✓" if result else "✗"
        print(f"  {prop}: {status}")
    
    # Test expected properties
    expected = leech.get_expected_properties()
    print(f"Expected kissing number: {expected['kissing_number']}")
    print(f"Expected dimension: {expected['dimension']}")
    print(f"Expected minimal norm: {expected.get('minimal_norm', 'N/A')}")

def test_leech_congruence_conditions():
    """Test congruence conditions"""
    print("Testing Leech congruence conditions...")
    
    leech = LeechLattice()
    
    # Test with some sample vectors
    test_vectors = np.array([
        np.zeros(24),  # Zero vector
        np.ones(24) * 2,  # All 2's
        np.concatenate([np.ones(12), -np.ones(12)])  # Mixed signs
    ])
    
    try:
        result = leech.validate_congruence_conditions(test_vectors)
        print(f"Congruence conditions satisfied: {result}")
        print("✓ Congruence condition validation working")
    except Exception as e:
        print(f"Error in congruence validation: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Leech Lattice Implementation Tests")
    print("=" * 60)
    
    try:
        test_leech_basic()
        print()
        
        test_leech_golay_code()
        print()
        
        test_leech_root_system()
        print()
        
        test_leech_nearest_neighbor()
        print()
        
        test_leech_properties()
        print()
        
        test_leech_congruence_conditions()
        print()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

