import numpy as np
from e8leech_project.lattices.e8_lattice import E8Lattice
from e8leech_project.lattices.leech_lattice import LeechLattice

def test_e8_theta_function_coefficients():
    e8_lattice = E8Lattice()
    expected_coeffs = {0: 1, 2: 240, 4: 2160, 6: 6720}
    computed_coeffs = e8_lattice.compute_theta_series_coefficients(max_n=6)
    for n, expected_val in expected_coeffs.items():
        assert computed_coeffs[n] == expected_val
    for n in [1, 3, 5]:
        assert computed_coeffs[n] == 0

def test_leech_theta_function_coefficients():
    leech_lattice = LeechLattice()
    expected_coeffs = {0: 1, 2: 0, 4: 196560}
    computed_coeffs = leech_lattice.compute_theta_series_coefficients(max_n=4)
    for n, expected_val in expected_coeffs.items():
        assert computed_coeffs[n] == expected_val
    for n in [1, 3]:
        assert computed_coeffs[n] == 0
