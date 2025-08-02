
import numpy as np
from e8leech.lattices.e8_lattice import E8Lattice
from e8leech.lattices.leech_lattice import LeechLattice

def test_e8_theta_function_coefficients():
    """
    Test E8 theta function coefficients against known values.
    """
    e8_lattice = E8Lattice()
    # Known coefficients for E8 theta function: 1 + 240q^2 + 2160q^4 + 6720q^6 + ...
    # The coefficients are for q^n, where n is the squared norm.
    # So, coefficients[0] is for q^0, coefficients[1] for q^1, etc.
    # E8 has no vectors of squared norm 1, 3, 5, etc.
    expected_coeffs = {
        0: 1,    # Constant term
        2: 240,  # Number of roots (squared norm 2)
        4: 2160, # Number of vectors with squared norm 4
        6: 6720, # Number of vectors with squared norm 6
    }

    # Compute up to a certain max_n to check several coefficients
    max_n_to_check = 6
    computed_coeffs = e8_lattice.compute_theta_series_coefficients(max_n=max_n_to_check)

    for n, expected_val in expected_coeffs.items():
        if n <= max_n_to_check:
            assert computed_coeffs[n] == expected_val, f"E8 theta coefficient for q^{n} mismatch: Expected {expected_val}, Got {computed_coeffs[n]}"

    # Check that coefficients for odd squared norms are zero
    for n in [1, 3, 5]:
        if n <= max_n_to_check:
            assert computed_coeffs[n] == 0, f"E8 theta coefficient for q^{n} should be 0, Got {computed_coeffs[n]}"

def test_leech_theta_function_coefficients():
    """
    Test Leech lattice theta function coefficients against known values.
    The Leech lattice has no vectors of squared norm 2.
    The shortest non-zero vectors have squared norm 4.
    """
    leech_lattice = LeechLattice()
    # Known coefficients for Leech lattice theta function:
    # 1 + 196560q^4 + ...
    expected_coeffs = {
        0: 1,      # Constant term
        2: 0,      # No vectors of squared norm 2
        4: 196560, # Kissing number (shortest non-zero vectors have squared norm 4)
    }

    max_n_to_check = 4
    computed_coeffs = leech_lattice.compute_theta_series_coefficients(max_n=max_n_to_check)

    for n, expected_val in expected_coeffs.items():
        if n <= max_n_to_check:
            assert computed_coeffs[n] == expected_val, f"Leech theta coefficient for q^{n} mismatch: Expected {expected_val}, Got {computed_coeffs[n]}"

    # Check that coefficients for odd squared norms are zero
    for n in [1, 3]:
        if n <= max_n_to_check:
            assert computed_coeffs[n] == 0, f"Leech theta coefficient for q^{n} should be 0, Got {computed_coeffs[n]}"



