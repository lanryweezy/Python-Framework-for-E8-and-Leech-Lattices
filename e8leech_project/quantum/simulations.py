"""
Quantum Computing Interfaces for lattice-based simulations.

This module provides integrations with Qiskit and Cirq for running
quantum simulations related to lattice problems, such as quantum attacks
on lattice-based cryptography or simulations of quantum field theories
on lattice structures.
"""

import numpy as np
from typing import Optional, Dict, Any

# Qiskit integration
try:
    from qiskit import QuantumCircuit, transpile, assemble
    from qiskit_aer import Aer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Cirq integration
try:
    import cirq
    CIRK_AVAILABLE = True
except ImportError:
    CIRK_AVAILABLE = False

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class QuantumSimulator:
    """
    Quantum simulator for lattice-based problems using Qiskit or Cirq.
    """

    def __init__(self, backend: str = "qiskit", num_qubits: int = 4):
        """
        Initialize the quantum simulator.

        Args:
            backend: The quantum computing backend to use ("qiskit" or "cirq").
            num_qubits: The number of qubits for the simulation.
        """
        if backend == "qiskit" and not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not installed. Please install it with 'pip install qiskit'.")
        if backend == "cirq" and not CIRK_AVAILABLE:
            raise ImportError("Cirq is not installed. Please install it with 'pip install cirq'.")

        self.backend = backend
        self.num_qubits = num_qubits
        self.circuit = self._create_circuit()

        logger.info(f"Initialized QuantumSimulator with backend='{backend}' and {num_qubits} qubits.")

    def _create_circuit(self):
        """
        Creates a quantum circuit based on the chosen backend.
        """
        if self.backend == "qiskit":
            return QuantumCircuit(self.num_qubits)
        elif self.backend == "cirq":
            return cirq.Circuit()

    def add_gate(self, gate_name: str, *qubits: int, **params: Any):
        """
        Adds a quantum gate to the circuit.

        Args:
            gate_name: The name of the gate (e.g., "h", "cx", "rz").
            qubits: The qubit(s) to apply the gate to.
            params: Parameters for the gate (e.g., angle for rotation gates).
        """
        if self.backend == "qiskit":
            self._add_qiskit_gate(gate_name, *qubits, **params)
        elif self.backend == "cirq":
            self._add_cirq_gate(gate_name, *qubits, **params)

    def _add_qiskit_gate(self, gate_name: str, *qubits: int, **params: Any):
        """Adds a gate to a Qiskit circuit."""
        gate_map = {
            "h": self.circuit.h,
            "cx": self.circuit.cx,
            "rz": self.circuit.rz,
            "measure": self.circuit.measure_all,
        }
        if gate_name in gate_map:
            if gate_name == "measure":
                gate_map[gate_name]()
            else:
                gate_map[gate_name](*qubits, **params)
        else:
            raise ValueError(f"Gate '{gate_name}' not supported for Qiskit backend.")

    def _add_cirq_gate(self, gate_name: str, *qubits: int, **params: Any):
        """Adds a gate to a Cirq circuit."""
        cirq_qubits = [cirq.LineQubit(i) for i in qubits]
        gate_map = {
            "h": cirq.H,
            "cx": cirq.CNOT,
            "rz": cirq.rz,
        }
        if gate_name in gate_map:
            self.circuit.append(gate_map[gate_name](*cirq_qubits, **params))
        elif gate_name == "measure":
            self.circuit.append(cirq.measure(*[cirq.LineQubit(i) for i in range(self.num_qubits)], key='result'))
        else:
            raise ValueError(f"Gate '{gate_name}' not supported for Cirq backend.")

    def run_simulation(self, shots: int = 1024) -> Dict[str, int]:
        """
        Runs the quantum simulation.

        Args:
            shots: The number of times to run the simulation.

        Returns:
            A dictionary of measurement results and their counts.
        """
        if self.backend == "qiskit":
            return self._run_qiskit_simulation(shots)
        elif self.backend == "cirq":
            return self._run_cirq_simulation(shots)

    def _run_qiskit_simulation(self, shots: int) -> Dict[str, int]:
        """Runs a Qiskit simulation."""
        self.circuit.measure_all()
        simulator = Aer.get_backend('qasm_simulator')
        compiled_circuit = transpile(self.circuit, simulator)
        qobj = assemble(compiled_circuit, shots=shots)
        result = simulator.run(qobj).result()
        counts = result.get_counts(self.circuit)
        return counts

    def _run_cirq_simulation(self, shots: int) -> Dict[str, int]:
        """Runs a Cirq simulation."""
        self.circuit.append(cirq.measure(*[cirq.LineQubit(i) for i in range(self.num_qubits)], key='result'))
        simulator = cirq.Simulator()
        result = simulator.run(self.circuit, repetitions=shots)
        hist = result.histogram(key='result')
        # Convert histogram to a format similar to Qiskit's counts
        counts = {bin(k)[2:].zfill(self.num_qubits): v for k, v in hist.items()}
        return counts

# Example usage
if __name__ == "__main__":
    # Qiskit Example
    if QISKIT_AVAILABLE:
        print("--- Running Qiskit Example ---")
        qiskit_sim = QuantumSimulator(backend="qiskit", num_qubits=2)
        qiskit_sim.add_gate("h", 0)
        qiskit_sim.add_gate("cx", 0, 1)
        qiskit_results = qiskit_sim.run_simulation()
        print("Qiskit results:", qiskit_results)
    else:
        print("Qiskit not available, skipping example.")

    # Cirq Example
    if CIRK_AVAILABLE:
        print("\n--- Running Cirq Example ---")
        cirq_sim = QuantumSimulator(backend="cirq", num_qubits=2)
        cirq_sim.add_gate("h", 0)
        cirq_sim.add_gate("cx", 0, 1)
        cirq_results = cirq_sim.run_simulation()
        print("Cirq results:", cirq_results)
    else:
        print("Cirq not available, skipping example.")
