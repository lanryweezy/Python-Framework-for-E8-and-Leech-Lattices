import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from e8leech_project.lattices.e8_lattice import E8Lattice

e8_lattice = E8Lattice()

@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(st.lists(st.integers(min_value=-5, max_value=5), min_size=8, max_size=8))
def test_e8_root_vector_norm(coords):
    roots = e8_lattice.generate_root_system()
    for root in roots:
        squared_norm = np.dot(root, root)
        assert np.isclose(squared_norm, 2.0)

@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=8, max_size=8))
def test_babai_nearest_plane_output_is_lattice_point(target_point):
    nearest_point = e8_lattice.nearest_neighbor(np.array(target_point), algorithm="babai")
    is_integer_or_half_integer = np.all(np.isclose(nearest_point * 2, np.round(nearest_point * 2)))
    assert is_integer_or_half_integer
    sum_coords = np.sum(nearest_point)
    assert np.isclose(sum_coords % 2, 0.0)

@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(st.lists(st.integers(min_value=-5, max_value=5), min_size=8, max_size=8))
def test_e8_even_lattice_property(coords):
    coeffs = np.array(coords)
    lattice_point = np.dot(coeffs, e8_lattice.basis)
    squared_norm = np.dot(lattice_point, lattice_point)
    assert np.isclose(squared_norm % 2, 0.0)

@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=8, max_size=8))
def test_lattice_quantization_output_is_lattice_point(target_point):
    quantized_point = e8_lattice.quantize(np.array([target_point]), algorithm="babai")[0]
    is_integer_or_half_integer = np.all(np.isclose(quantized_point * 2, np.round(quantized_point * 2)))
    assert is_integer_or_half_integer
    sum_coords = np.sum(quantized_point)
    assert np.isclose(sum_coords % 2, 0.0)
