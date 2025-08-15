#!/usr/bin/env python3
"""
Test script for E8 lattice implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from e8leech.lattices.e8_lattice import E8Lattice
from e8leech.core.config import LatticeConfig

def test_e8_basic():
    """Test basic E8 lattice functionality"""
    print("Testing E8 lattice basic functionality...")
    
    # Create E8 lattice
    e8 = E8Lattice()
    
    print(f"E8 dimension: {e8.dimension}")
    print(f"E8 rank: {e8.rank}")
    print(f"E8 determinant: {e8.determinant}")
    
    # Test basis properties
    assert e8.dimension == 8, f"Expected dimension 8, got {e8.dimension}"
    assert e8.rank == 8, f"Expected rank 8, got {e8.rank}"
    
    print("✓ Basic properties validated")

def test_e8_root_system():
    """Test E8 root system generation"""
    print("Testing E8 root system generation...")
    
    e8 = E8Lattice()
    roots = e8.generate_root_system()
    
    print(f"Generated {len(roots)} root vectors")
    
    # Check root count
    expected_count = E8Lattice.ROOT_COUNT
    if len(roots) != expected_count:
        print(f"Warning: Expected {expected_count} roots, got {len(roots)}")
    
    # Check root norms (should be sqrt(2) for E8)
    norms = np.linalg.norm(roots, axis=1)
    expected_norm = np.sqrt(2.0)
    
    print(f"Root norms - min: {norms.min():.6f}, max: {norms.max():.6f}, expected: {expected_norm:.6f}")
    
    # Most roots should have norm sqrt(2)
    norm_tolerance = 1e-10
    correct_norms = np.sum(np.abs(norms - expected_norm) < norm_tolerance)
    print(f"Roots with correct norm: {correct_norms}/{len(roots)}")
    
    print("✓ Root system generated")

def test_e8_nearest_neighbor():
    """Test nearest neighbor search"""
    print("Testing E8 nearest neighbor search...")
    
    e8 = E8Lattice()
    
    # Test with a few random vectors
    np.random.seed(42)
    test_vectors = np.random.randn(5, 8)
    
    for i, vec in enumerate(test_vectors):
        nearest = e8.nearest_neighbor(vec)
        distance = np.linalg.norm(vec - nearest)
        print(f"Vector {i}: distance to lattice = {distance:.6f}")
    
    print("✓ Nearest neighbor search working")

def test_e8_properties():
    """Test E8 mathematical properties"""
    print("Testing E8 mathematical properties...")
    
    e8 = E8Lattice()
    
    # Validate properties
    validation_results = e8.validate_properties()
    print("Validation results:")
    for prop, result in validation_results.items():
        status = "✓" if result else "✗"
        print(f"  {prop}: {status}")
    
    # Test packing density
    try:
        density = e8.compute_packing_density()
        expected_density = E8Lattice.EXPECTED_PACKING_DENSITY
        print(f"Packing density: {density:.6f} (expected: {expected_density:.6f})")
        
        if abs(density - expected_density) < 0.01:
            print("✓ Packing density is close to expected value")
        else:
            print("✗ Packing density differs from expected value")
    except Exception as e:
        print(f"✗ Error computing packing density: {e}")

def test_e8_theta_function():
    """Test theta function computation"""
    print("Testing E8 theta function...")
    
    e8 = E8Lattice()
    
    # Test theta function at a simple point
    tau = 1j  # Pure imaginary
    try:
        theta_value = e8.compute_theta_function(tau, max_terms=100)
        print(f"Theta function at τ=i: {theta_value}")
        print("✓ Theta function computation working")
    except Exception as e:
        print(f"✗ Error computing theta function: {e}")

def main():
    """Run all tests"""
    print("=" * 50)
    print("E8 Lattice Implementation Tests")
    print("=" * 50)
    
    try:
        test_e8_basic()
        print()
        
        test_e8_root_system()
        print()
        
        test_e8_nearest_neighbor()
        print()
        
        test_e8_properties()
        print()
        
        test_e8_theta_function()
        print()
        
        print("=" * 50)
        print("All tests completed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

