import pytest
import time
import numpy as np
from e8leech_project.lattices.e8_lattice import E8Lattice
from e8leech_project.lattices.leech_lattice import LeechLattice
from e8leech_project.crypto.lwe import LWE
from e8leech_project.ai.equivariant_gnn import EquivariantGNN

class MockSageMathE8:
    def generate_root_system(self):
        return np.random.rand(240, 8)
    def babai_nearest_plane(self, point):
        return np.zeros(8)

class MockNISTPQC:
    def encrypt(self, plaintext):
        time.sleep(0.001)
        return b'ciphertext'
    def decrypt(self, ciphertext):
        time.sleep(0.001)
        return b'plaintext'

class MockClassicalML:
    def train(self, data):
        time.sleep(0.01)
    def predict(self, data):
        time.sleep(0.001)

@pytest.fixture
def e8_lattice():
    return E8Lattice()

@pytest.fixture
def leech_lattice():
    return LeechLattice()

@pytest.fixture
def lwe_crypto():
    return LWE(dimension=24, modulus=257)

@pytest.fixture
def equivariant_gnn():
    return EquivariantGNN(input_dim=24, hidden_dim=64, output_dim=24, num_layers=2)

class TestPerformanceBenchmarks:
    def test_e8_root_system_generation_benchmark(self, e8_lattice):
        e8_lattice.generate_root_system()

    def test_e8_babai_nearest_plane_benchmark(self, e8_lattice):
        e8_lattice.nearest_neighbor(np.random.rand(8), algorithm="babai")

    def test_leech_babai_nearest_plane_benchmark(self, leech_lattice):
        leech_lattice.nearest_neighbor(np.random.rand(24), algorithm="babai")

    def test_lwe_encryption_decryption_benchmark(self, lwe_crypto):
        plaintext = b'test_message'
        ciphertext = lwe_crypto.encrypt(plaintext)
        decrypted_text = lwe_crypto.decrypt(ciphertext)
        assert plaintext == decrypted_text

    def test_ai_model_training_benchmark(self, equivariant_gnn):
        equivariant_gnn.train(np.random.rand(100, 24))

    def test_ai_model_prediction_benchmark(self, equivariant_gnn):
        equivariant_gnn.predict(np.random.rand(10, 24))
