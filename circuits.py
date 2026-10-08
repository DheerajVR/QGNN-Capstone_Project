"""
Circuit corpus: the "specific circuits" whose per-qubit error we predict.

Every builder returns a *logical* k-qubit circuit with no measurements. The
dataset stage transpiles each one onto a concrete patch of a real device.

Families are deliberately diverse in the two things that drive hardware error:
how much two-qubit entangling work each qubit does, and how long each qubit
sits idle while other qubits are busy.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import random_clifford, random_unitary

FAMILIES = [
    "ghz",
    "w_state",
    "qft",
    "bernstein_vazirani",
    "quantum_volume",
    "random_clifford",
    "trotter_ising",
    "hardware_efficient_ansatz",
    "qaoa_maxcut",
    "grover",
    "ripple_adder",
    "phase_estimation",
]


def ghz(k, rng, depth=1):
    qc = QuantumCircuit(k)
    for _ in range(depth):
        qc.h(0)
        for i in range(k - 1):
            qc.cx(i, i + 1)
    return qc


def w_state(k, rng, depth=1):
    """Approximate W-state prep via a cascade of controlled rotations."""
    qc = QuantumCircuit(k)
    qc.x(0)
    for i in range(k - 1):
        theta = 2 * np.arccos(np.sqrt(1.0 / (k - i)))
        qc.ry(theta / 2, i + 1)
        qc.cz(i, i + 1)
        qc.ry(-theta / 2, i + 1)
        qc.cx(i + 1, i)
    return qc


def qft(k, rng, depth=1):
    qc = QuantumCircuit(k)
    bits = rng.integers(0, 2, k)
    for i, b in enumerate(bits):
        if b:
            qc.x(i)
    qc.compose(QFT(k, do_swaps=True).decompose(), inplace=True)
    return qc


def bernstein_vazirani(k, rng, depth=1):
    n = k - 1
    secret = rng.integers(0, 2, n)
    qc = QuantumCircuit(k)
    qc.x(k - 1)
    qc.h(range(k))
    for i, b in enumerate(secret):
        if b:
            qc.cx(i, k - 1)
    qc.h(range(n))
    return qc


def quantum_volume(k, rng, depth=None):
    """QV-style: layers of random SU(4) on random disjoint pairs."""
    d = depth or k
    qc = QuantumCircuit(k)
    for _ in range(d):
        perm = rng.permutation(k)
        for a in range(0, k - 1, 2):
            q0, q1 = int(perm[a]), int(perm[a + 1])
            qc.unitary(random_unitary(4, seed=int(rng.integers(1 << 30))), [q0, q1])
    return qc


def random_clifford_circ(k, rng, depth=1):
    qc = QuantumCircuit(k)
    qc.compose(random_clifford(k, seed=int(rng.integers(1 << 30))).to_circuit(), inplace=True)
    return qc


def trotter_ising(k, rng, depth=3):
    """Trotterised transverse-field Ising evolution on a line."""
    qc = QuantumCircuit(k)
    dt = 0.35
    qc.h(range(k))
    for _ in range(depth):
        for i in range(0, k - 1):
            qc.rzz(2 * dt, i, i + 1)
        for i in range(k):
            qc.rx(2 * dt, i)
    return qc


def hardware_efficient_ansatz(k, rng, depth=3):
    """The workhorse VQE ansatz: Ry/Rz layer + entangling ladder."""
    qc = QuantumCircuit(k)
    for _ in range(depth):
        for i in range(k):
            qc.ry(float(rng.uniform(0, 2 * np.pi)), i)
            qc.rz(float(rng.uniform(0, 2 * np.pi)), i)
        for i in range(k - 1):
            qc.cx(i, i + 1)
    for i in range(k):
        qc.ry(float(rng.uniform(0, 2 * np.pi)), i)
    return qc


def qaoa_maxcut(k, rng, depth=2):
    edges = [(i, j) for i in range(k) for j in range(i + 1, k) if rng.random() < 0.45]
    if not edges:
        edges = [(i, i + 1) for i in range(k - 1)]
    qc = QuantumCircuit(k)
    qc.h(range(k))
    for p in range(depth):
        gamma = float(rng.uniform(0.2, 1.2))
        beta = float(rng.uniform(0.2, 1.2))
        for (u, v) in edges:
            qc.cx(u, v)
            qc.rz(2 * gamma, v)
            qc.cx(u, v)
        for i in range(k):
            qc.rx(2 * beta, i)
    return qc


def grover(k, rng, depth=1):
    """Grover with a multi-controlled-Z oracle on a random marked string."""
    n = k
    marked = rng.integers(0, 2, n)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    iters = max(1, int(depth))
    for _ in range(iters):
        for i, b in enumerate(marked):
            if not b:
                qc.x(i)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for i, b in enumerate(marked):
            if not b:
                qc.x(i)
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        if n > 1:
            qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))
    return qc


def ripple_adder(k, rng, depth=1):
    """Cuccaro-style ripple-carry addition on floor(k/2) bit operands."""
    n = max(1, (k - 1) // 2)
    qc = QuantumCircuit(k)
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    for i in range(n):
        if a[i]:
            qc.x(i)
        if b[i]:
            qc.x(n + i)
    carry = 2 * n
    for i in range(n):
        if 2 * n < k:
            qc.ccx(i, n + i, carry)
        qc.cx(i, n + i)
    for i in reversed(range(n)):
        qc.cx(i, n + i)
        if 2 * n < k:
            qc.ccx(i, n + i, carry)
    return qc


def phase_estimation(k, rng, depth=1):
    """QPE for a random phase, k-1 counting qubits."""
    n = k - 1
    phase = float(rng.uniform(0.05, 0.95))
    qc = QuantumCircuit(k)
    qc.x(k - 1)
    qc.h(range(n))
    for i in range(n):
        for _ in range(2 ** i):
            qc.cp(2 * np.pi * phase, i, k - 1)
    qc.compose(QFT(n, do_swaps=True, inverse=True).decompose(), qubits=range(n), inplace=True)
    return qc


_BUILDERS = {
    "ghz": ghz,
    "w_state": w_state,
    "qft": qft,
    "bernstein_vazirani": bernstein_vazirani,
    "quantum_volume": quantum_volume,
    "random_clifford": random_clifford_circ,
    "trotter_ising": trotter_ising,
    "hardware_efficient_ansatz": hardware_efficient_ansatz,
    "qaoa_maxcut": qaoa_maxcut,
    "grover": grover,
    "ripple_adder": ripple_adder,
    "phase_estimation": phase_estimation,
}

# Depth ranges chosen so each family spans easy -> hard on real hardware.
_DEPTH_RANGE = {
    "ghz": (1, 2),
    "w_state": (1, 1),
    "qft": (1, 1),
    "bernstein_vazirani": (1, 1),
    "quantum_volume": (2, 8),
    "random_clifford": (1, 1),
    "trotter_ising": (1, 6),
    "hardware_efficient_ansatz": (1, 5),
    "qaoa_maxcut": (1, 3),
    "grover": (1, 2),
    "ripple_adder": (1, 1),
    "phase_estimation": (1, 1),
}


def sample_circuit(family: str, k: int, rng: np.random.Generator):
    lo, hi = _DEPTH_RANGE[family]
    depth = int(rng.integers(lo, hi + 1))
    qc = _BUILDERS[family](k, rng, depth)
    qc.name = f"{family}_k{k}_d{depth}"
    return qc, depth
