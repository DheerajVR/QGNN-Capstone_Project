"""
Mapping a logical circuit onto a real device patch, and computing ground truth.

Pipeline for one sample:

    patch      = sample_patch(cal, k)              # connected qubits on the real chip
    sub_target = build_sub_target(cal, patch)      # real T1/T2/gate-error/duration
    tqc        = transpile_to_patch(logical, sub_target)
    sched      = asap_schedule(tqc, sub_target)    # explicit Delay ops -> real idle time
    truth      = ground_truth(sched, sub_target, cal, patch)

Ground truth is computed by exact density-matrix simulation of the patch under
the device's own error channels (Aer's `basic_device_gate_errors`, i.e. IBM's
own calibration -> channel conversion), plus the measured readout confusion
matrices applied analytically. No shot noise, so labels are exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Delay, Parameter
from qiskit.circuit.library import CXGate, CZGate, ECRGate, IGate, Measure, RZGate, SXGate, XGate
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit.providers import QubitProperties
from qiskit.transpiler import InstructionProperties, Target
from qiskit_aer import AerSimulator
from qiskit_aer.noise.device import basic_device_gate_errors, basic_device_readout_errors
from qiskit_aer.noise.errors import thermal_relaxation_error

_TWOQ_GATE_OBJ = {"ecr": ECRGate(), "cz": CZGate(), "cx": CXGate()}


# ---------------------------------------------------------------------------
# 1. Pick a connected patch of physical qubits
# ---------------------------------------------------------------------------

def sample_patch(cal, k: int, rng: np.random.Generator, bias: str = "random") -> list[int]:
    """
    Randomised BFS from a seed qubit, returning `k` connected physical qubits.

    `bias` steers which part of the chip we land on so the dataset covers good,
    typical and bad hardware regions:
      'random' - uniform seed, random frontier expansion
      'good'   - prefer low-error edges (what a smart transpiler would pick)
      'bad'    - prefer high-error edges (stress cases)
    """
    G = cal.graph
    nodes = [n for n in G.nodes if G.degree(n) > 0]
    if len(nodes) < k:
        return []
    for _ in range(40):
        seed = int(rng.choice(nodes))
        chosen = [seed]
        frontier = list(G.neighbors(seed))
        while len(chosen) < k and frontier:
            frontier = [f for f in frontier if f not in chosen]
            if not frontier:
                break
            if bias == "random":
                pick = int(rng.choice(frontier))
            else:
                errs = np.array(
                    [min(G[f][c]["err_2q"] for c in chosen if G.has_edge(f, c)) for f in frontier]
                )
                order = np.argsort(errs if bias == "good" else -errs)
                # softly random among the top few so patches are not deterministic
                top = order[: max(1, len(order) // 2)]
                pick = int(frontier[int(rng.choice(top))])
            chosen.append(pick)
            frontier.extend(G.neighbors(pick))
        if len(chosen) == k:
            return chosen
    return []


# ---------------------------------------------------------------------------
# 2. Build a Qiskit Target for that patch, carrying the real calibration
# ---------------------------------------------------------------------------

def build_sub_target(cal, patch: list[int]) -> Target:
    """Target on len(patch) qubits, indices local, properties copied from the real chip."""
    k = len(patch)
    loc = {p: i for i, p in enumerate(patch)}
    q = cal.qubits

    tgt = Target(num_qubits=k)
    tgt.qubit_properties = [
        QubitProperties(t1=float(q.at[p, "t1"]), t2=float(q.at[p, "t2"]),
                        frequency=float(q.at[p, "frequency"]))
        for p in patch
    ]

    one_q = str(q.at[patch[0], "one_q_gate"])
    props_1q = {
        (loc[p],): InstructionProperties(duration=float(q.at[p, "dur_1q"]),
                                         error=float(q.at[p, "err_1q"]))
        for p in patch
    }
    gate_obj = SXGate() if one_q == "sx" else XGate()
    tgt.add_instruction(gate_obj, props_1q)
    other = XGate() if one_q == "sx" else SXGate()
    tgt.add_instruction(other, dict(props_1q))
    tgt.add_instruction(IGate(), {(loc[p],): InstructionProperties(duration=0.0, error=0.0) for p in patch})
    tgt.add_instruction(RZGate(Parameter("theta")),
                        {(loc[p],): InstructionProperties(duration=0.0, error=0.0) for p in patch})

    twoq_props = {}
    for r in cal.edges.itertuples():
        u, v = int(r.control), int(r.target)
        if u in loc and v in loc:
            twoq_props[(loc[u], loc[v])] = InstructionProperties(
                duration=float(r.dur_2q), error=float(r.err_2q)
            )
    if not twoq_props:
        raise ValueError("patch has no usable two-qubit couplings")
    tgt.add_instruction(_TWOQ_GATE_OBJ[cal.two_q_gate], twoq_props)

    tgt.add_instruction(
        Measure(),
        {(loc[p],): InstructionProperties(duration=float(q.at[p, "readout_duration"]),
                                          error=float(q.at[p, "readout_error"]))
         for p in patch},
    )
    tgt.add_instruction(Delay(Parameter("t")), {(loc[p],): None for p in patch})
    return tgt


def _connected(target: Target, k: int) -> bool:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(k))
    for name in target.operation_names:
        if name in _TWOQ_GATE_OBJ:
            for pair in target[name]:
                g.add_edge(*pair)
    return nx.is_connected(g) if k > 1 else True


# ---------------------------------------------------------------------------
# 3. Transpile + ASAP schedule with explicit idle delays
# ---------------------------------------------------------------------------

def transpile_to_patch(logical: QuantumCircuit, sub_target: Target, seed: int,
                       optimization_level: int = 1) -> Optional[QuantumCircuit]:
    try:
        return transpile(logical, target=sub_target, seed_transpiler=seed,
                         optimization_level=optimization_level)
    except Exception:
        return None


def asap_schedule(tqc: QuantumCircuit, sub_target: Target):
    """
    ASAP-schedule the transpiled circuit and materialise idle periods as Delay
    instructions, so decoherence while a qubit waits is physically simulated.

    Returns (scheduled_circuit, per_qubit_stats).
    """
    k = tqc.num_qubits
    clock = np.zeros(k)
    busy = np.zeros(k)
    n1q = np.zeros(k, dtype=int)
    n2q = np.zeros(k, dtype=int)
    first_touch = np.full(k, np.inf)
    last_touch = np.zeros(k)

    sched = QuantumCircuit(k)
    for inst in tqc.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qidx = [tqc.find_bit(qb).index for qb in inst.qubits]
        try:
            dur = sub_target[name][tuple(qidx)].duration or 0.0
        except Exception:
            dur = 0.0
        start = max(clock[i] for i in qidx)
        for i in qidx:
            gap = start - clock[i]
            if gap > 1e-12:
                sched.delay(gap, i, unit="s")
            clock[i] = start + dur
            busy[i] += dur
            first_touch[i] = min(first_touch[i], start)
            last_touch[i] = max(last_touch[i], start + dur)
        sched.append(inst.operation, [sched.qubits[i] for i in qidx])
        if len(qidx) == 1 and dur > 0:
            n1q[qidx[0]] += 1
        elif len(qidx) == 2:
            n2q[qidx[0]] += 1
            n2q[qidx[1]] += 1

    total = float(clock.max()) if k else 0.0
    for i in range(k):
        gap = total - clock[i]
        if gap > 1e-12:
            sched.delay(gap, i, unit="s")

    first_touch[np.isinf(first_touch)] = 0.0
    stats = dict(
        n_1q=n1q,
        n_2q=n2q,
        busy=busy,
        idle=np.maximum(total - busy, 0.0),
        span=np.maximum(last_touch - first_touch, 0.0),
        total_duration=total,
    )
    return sched, stats


# ---------------------------------------------------------------------------
# 4. Insert the device's own error channels and simulate
# ---------------------------------------------------------------------------

def _gate_error_map(sub_target: Target) -> dict:
    out = {}
    for name, qubits, err in basic_device_gate_errors(target=sub_target):
        out[(name, tuple(qubits))] = err
    return out


def _readout_matrices(sub_target: Target, k: int) -> np.ndarray:
    """A[q] is the 2x2 confusion matrix A[m, t] = P(measure m | true t)."""
    A = np.tile(np.eye(2), (k, 1, 1))
    for qubits, ro in basic_device_readout_errors(target=sub_target):
        q = qubits[0]
        # ReadoutError.probabilities[t][m] = P(measure m | true t) -> transpose
        A[q] = np.asarray(ro.probabilities).T
    return A


def build_noisy_circuit(sched: QuantumCircuit, sub_target: Target, err_map: dict) -> QuantumCircuit:
    """Copy of `sched` with the device's Kraus channels inserted inline."""
    k = sched.num_qubits
    qp = sub_target.qubit_properties
    out = QuantumCircuit(k)
    for inst in sched.data:
        name = inst.operation.name
        qidx = [sched.find_bit(qb).index for qb in inst.qubits]
        if name == "delay":
            t = float(inst.operation.duration)
            i = qidx[0]
            if t > 0:
                ch = thermal_relaxation_error(qp[i].t1, qp[i].t2, t)
                out.append(ch.to_instruction(), [out.qubits[i]])
            continue
        out.append(inst.operation, [out.qubits[i] for i in qidx])
        err = err_map.get((name, tuple(qidx)))
        if err is not None:
            out.append(err.to_instruction(), [out.qubits[i] for i in qidx])
    return out


def _strip_delays(qc: QuantumCircuit) -> QuantumCircuit:
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name in ("delay", "barrier"):
            continue
        out.append(inst.operation, [out.qubits[qc.find_bit(q).index] for q in inst.qubits])
    return out


def _apply_readout(probs: np.ndarray, A: np.ndarray, k: int) -> np.ndarray:
    """Apply per-qubit confusion matrices to a 2**k probability vector."""
    t = probs.reshape((2,) * k)  # axis a <-> qubit (k-1-a) in Qiskit little-endian
    for a in range(k):
        q = k - 1 - a
        t = np.tensordot(A[q], t, axes=([1], [a]))
        t = np.moveaxis(t, 0, a)
    return t.reshape(-1)


@dataclass
class GroundTruth:
    probs_ideal: np.ndarray
    probs_noisy: np.ndarray
    q_infidelity: np.ndarray       # 1 - F(rho_q^noisy, rho_q^ideal), per qubit
    q_marginal_dev: np.ndarray     # |P_noisy(q=1) - P_ideal(q=1)|, per qubit
    circuit_fidelity: float        # Hellinger fidelity of output distributions
    circuit_tvd: float


_SIM = AerSimulator(method="density_matrix")


def ground_truth(sched: QuantumCircuit, sub_target: Target) -> Optional[GroundTruth]:
    k = sched.num_qubits
    err_map = _gate_error_map(sub_target)
    A = _readout_matrices(sub_target, k)

    clean = _strip_delays(sched)
    try:
        psi = Statevector.from_instruction(clean)
    except Exception:
        return None
    rho_ideal = DensityMatrix(psi)

    noisy = build_noisy_circuit(sched, sub_target, err_map)
    noisy.save_density_matrix(label="rho")
    try:
        res = _SIM.run(noisy).result()
        rho_noisy = DensityMatrix(np.asarray(res.data()["rho"]))
    except Exception:
        return None

    p_ideal = np.clip(np.real(np.diag(rho_ideal.data)), 0, None)
    p_noisy = np.clip(np.real(np.diag(rho_noisy.data)), 0, None)
    p_ideal /= p_ideal.sum()
    p_noisy /= p_noisy.sum()
    p_ideal_m = _apply_readout(p_ideal, A, k)
    p_noisy_m = _apply_readout(p_noisy, A, k)

    q_inf = np.zeros(k)
    q_dev = np.zeros(k)
    for q in range(k):
        keep = [i for i in range(k) if i != q]
        ri = partial_trace(rho_ideal, keep).data if k > 1 else rho_ideal.data
        rn = partial_trace(rho_noisy, keep).data if k > 1 else rho_noisy.data
        q_inf[q] = 1.0 - _fid_1q(rn, ri)
        # readout acts on the single-qubit marginal after tracing out the rest
        pi1 = float(np.real(ri[1, 1]))
        pn1 = float(np.real(rn[1, 1]))
        pi1 = A[q][1, 0] * (1 - pi1) + A[q][1, 1] * pi1
        pn1 = A[q][1, 0] * (1 - pn1) + A[q][1, 1] * pn1
        q_dev[q] = abs(pn1 - pi1)

    hell = float(np.sum(np.sqrt(p_ideal_m * p_noisy_m)) ** 2)
    tvd = float(0.5 * np.sum(np.abs(p_ideal_m - p_noisy_m)))
    return GroundTruth(
        probs_ideal=p_ideal_m,
        probs_noisy=p_noisy_m,
        q_infidelity=np.clip(q_inf, 0.0, 1.0),
        q_marginal_dev=q_dev,
        circuit_fidelity=float(np.clip(hell, 0.0, 1.0)),
        circuit_tvd=tvd,
    )


def _fid_1q(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity for 1-qubit states, closed form via Bloch vectors."""
    def bloch(m):
        return np.array([
            2 * np.real(m[0, 1]),
            2 * np.imag(m[1, 0]),
            np.real(m[0, 0] - m[1, 1]),
        ])

    r, s = bloch(rho), bloch(sigma)
    nr, ns = float(r @ r), float(s @ s)
    val = 0.5 * (1.0 + float(r @ s) + np.sqrt(max(0.0, (1 - nr) * (1 - ns))))
    return float(np.clip(val, 0.0, 1.0))
