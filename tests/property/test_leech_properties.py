import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from e8leech_project.lattices.leech_lattice import LeechLattice

leech_lattice = LeechLattice()

@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(st.lists(st.integers(min_value=-5, max_value=5), min_size=24, max_size=24))
def test_leech_unimodularity(coords):
    determinant = leech_lattice.determinant
    assert np.isclose(determinant, 1.0)

@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(st.lists(st.integers(min_value=-5, max_value=5), min_size=24, max_size=24))
def test_leech_even_lattice_property(coords):
    coeffs = np.array(coords)
    lattice_point = np.dot(coeffs, leech_lattice.basis)
    squared_norm = np.dot(lattice_point, lattice_point)
    assert np.isclose(squared_norm % 2, 0.0)

@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=24, max_size=24))
def test_leech_quantization_output_is_lattice_point(target_point):
    quantized_point = leech_lattice.quantize(np.array([target_point]), algorithm="babai")[0]
    is_integer_or_half_integer = np.all(np.isclose(quantized_point * 2, np.round(quantized_point * 2)))
    assert is_integer_or_half_integer
    sum_coords = np.sum(quantized_point)
    assert np.isclose(sum_coords % 2, 0.0)
