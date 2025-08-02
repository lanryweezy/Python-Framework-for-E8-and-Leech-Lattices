import pytest
import time
import numpy as np
from e8leech.lattices.e8_lattice import E8Lattice
from e8leech.lattices.leech_lattice import LeechLattice
from e8leech.modules.crypto.lwe import LWE
from e8leech.modules.ai.equivariant_gnn import EquivariantGNN

# Mock SageMath E8 for benchmarking comparison
class MockSageMathE8:
    def __init__(self):
        self.basis = np.eye(8)

    def generate_root_system(self):
        # Simulate SageMath's E8 root system generation
        return np.random.rand(240, 8) # Placeholder

    def babai_nearest_plane(self, point):
        # Simulate Babai's algorithm
        return np.zeros(8) # Placeholder

# Mock NIST PQC finalist (e.g., Kyber) for benchmarking comparison
class MockNISTPQC:
    def __init__(self):
        pass

    def encrypt(self, plaintext):
        time.sleep(0.001) # Simulate encryption time
        return b'ciphertext'

    def decrypt(self, ciphertext):
        time.sleep(0.001) # Simulate decryption time
        return b'plaintext'

# Mock Classical ML model for benchmarking comparison
class MockClassicalML:
    def __init__(self):
        pass

    def train(self, data):
        time.sleep(0.01) # Simulate training time

    def predict(self, data):
        time.sleep(0.001) # Simulate prediction time

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
        start_time = time.time()
        e8_lattice.generate_root_system()
        end_time = time.time()
        print(f"\nE8 Root System Generation: {end_time - start_time:.4f} seconds")

        mock_sage_e8 = MockSageMathE8()
        start_time = time.time()
        mock_sage_e8.generate_root_system()
        end_time = time.time()
        print(f"SageMath E8 Root System Generation (Mock): {end_time - start_time:.4f} seconds")

    def test_e8_babai_nearest_plane_benchmark(self, e8_lattice):
        point = np.random.rand(8)
        start_time = time.time()
        e8_lattice.nearest_neighbor(point, algorithm="babai")
        end_time = time.time()
        print(f"E8 Babai Nearest Plane: {end_time - start_time:.4f} seconds")

        mock_sage_e8 = MockSageMathE8()
        start_time = time.time()
        mock_sage_e8.babai_nearest_plane(point)
        end_time = time.time()
        print(f"SageMath E8 Babai Nearest Plane (Mock): {end_time - start_time:.4f} seconds")

    def test_leech_babai_nearest_plane_benchmark(self, leech_lattice):
        point = np.random.rand(24)
        start_time = time.time()
        leech_lattice.nearest_neighbor(point, algorithm="babai")
        end_time = time.time()
        print(f"Leech Babai Nearest Plane: {end_time - start_time:.4f} seconds")

    def test_lwe_encryption_decryption_benchmark(self, lwe_crypto):
        plaintext = b'test_message'
        start_time = time.time()
        ciphertext = lwe_crypto.encrypt(plaintext)
        decrypted_text = lwe_crypto.decrypt(ciphertext)
        end_time = time.time()
        print(f"LWE Encryption/Decryption: {end_time - start_time:.4f} seconds")
        assert plaintext == decrypted_text

        mock_nist_pqc = MockNISTPQC()
        start_time = time.time()
        ciphertext = mock_nist_pqc.encrypt(plaintext)
        decrypted_text = mock_nist_pqc.decrypt(ciphertext)
        end_time = time.time()
        print(f"NIST PQC (Mock) Encryption/Decryption: {end_time - start_time:.4f} seconds")

    def test_ai_model_training_benchmark(self, equivariant_gnn):
        data = np.random.rand(100, 24) # Simulate 100 lattice points
        start_time = time.time()
        equivariant_gnn.train(data)
        end_time = time.time()
        print(f"Equivariant GNN Training: {end_time - start_time:.4f} seconds")

        mock_classical_ml = MockClassicalML()
        start_time = time.time()
        mock_classical_ml.train(data)
        end_time = time.time()
        print(f"Classical ML (Mock) Training: {end_time - start_time:.4f} seconds")

    def test_ai_model_prediction_benchmark(self, equivariant_gnn):
        data = np.random.rand(10, 24) # Simulate 10 lattice points
        start_time = time.time()
        equivariant_gnn.predict(data)
        end_time = time.time()
        print(f"Equivariant GNN Prediction: {end_time - start_time:.4f} seconds")

        mock_classical_ml = MockClassicalML()
        start_time = time.time()
        mock_classical_ml.predict(data)
        end_time = time.time()
        print(f"Classical ML (Mock) Prediction: {end_time - start_time:.4f} seconds")


