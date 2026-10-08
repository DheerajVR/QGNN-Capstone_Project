"""
Real IBM Quantum device calibration ingestion.

Two sources, one schema:

1. `load_snapshot(name)`  -- real calibration snapshots shipped inside
   `qiskit_ibm_runtime.fake_provider`. These are *recorded from real IBM
   hardware* (T1, T2, qubit frequency, readout error, per-pair two-qubit gate
   error and duration) and are what we train on here.

2. `load_live(backend_name, token)` -- pulls the *current* calibration of a
   real backend from your own IBM Quantum account. Same schema out, so every
   downstream stage (graphs, model, evaluation) is identical.

A `Calibration` is deliberately plain (numpy / pandas / networkx) so nothing
downstream depends on Qiskit internals.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Two-qubit gate error above this is a failed/uncalibrated pair, not a usable edge.
MAX_USABLE_2Q_ERROR = 0.5
# Devices whose usable-edge fraction falls below this are dropped entirely.
MIN_USABLE_EDGE_FRACTION = 0.5

# Snapshots Qiskit itself flags as not representative of the real device.
NON_REPRESENTATIVE = {"FakeNighthawk"}

TWO_Q_GATES = ("ecr", "cz", "cx")
ONE_Q_GATES = ("sx", "x")

# Processor family, inferred from size + basis gate. Used only for reporting /
# stratified splits, never as a model feature.
FAMILY_BY_SIZE = {
    5: "Falcon-r4/r5",
    7: "Falcon-r5",
    15: "Melbourne",
    16: "Falcon-r4",
    20: "Penguin",
    27: "Falcon-r5",
    28: "Hummingbird",
    33: "Heron-r1",
    53: "Hummingbird-r3",
    65: "Hummingbird-r3",
    120: "Nighthawk",
    127: "Eagle-r3",
    133: "Heron-r1",
    156: "Heron-r2",
}


@dataclass
class Calibration:
    """Device-agnostic calibration snapshot."""

    name: str
    n_qubits: int
    two_q_gate: str
    qubits: pd.DataFrame  # index = physical qubit
    edges: pd.DataFrame  # one row per *directed* usable coupling
    graph: nx.Graph = field(repr=False)
    source: str = "snapshot"

    @property
    def family(self) -> str:
        return FAMILY_BY_SIZE.get(self.n_qubits, f"{self.n_qubits}q")

    def summary(self) -> dict:
        q, e = self.qubits, self.edges
        return {
            "device": self.name,
            "source": self.source,
            "family": self.family,
            "n_qubits": self.n_qubits,
            "two_q_gate": self.two_q_gate,
            "n_edges": self.graph.number_of_edges(),
            "median_T1_us": float(np.nanmedian(q["t1"]) * 1e6),
            "median_T2_us": float(np.nanmedian(q["t2"]) * 1e6),
            "median_readout_err": float(np.nanmedian(q["readout_error"])),
            "median_1q_err": float(np.nanmedian(q["err_1q"])),
            "median_2q_err": float(np.nanmedian(e["err_2q"])) if len(e) else float("nan"),
            "worst_2q_err": float(np.nanmax(e["err_2q"])) if len(e) else float("nan"),
        }


def _pick_two_q_gate(target) -> Optional[str]:
    for g in TWO_Q_GATES:
        if g in target.operation_names:
            return g
    return None


def calibration_from_target(target, name: str, source: str = "snapshot") -> Optional[Calibration]:
    """Convert a Qiskit `Target` (fake snapshot or live backend) into a `Calibration`."""
    n = target.num_qubits
    two_q = _pick_two_q_gate(target)
    if two_q is None or n < 5:
        return None

    # ---- per-qubit ----------------------------------------------------------
    qp = target.qubit_properties
    rows = []
    meas = target["measure"] if "measure" in target.operation_names else {}
    oneq = {}
    for g in ONE_Q_GATES:
        if g in target.operation_names:
            oneq = target[g]
            oneq_name = g
            break
    else:
        oneq_name = None

    for q in range(n):
        p = qp[q] if q < len(qp) else None
        t1 = getattr(p, "t1", None)
        t2 = getattr(p, "t2", None)
        freq = getattr(p, "frequency", None)
        m = meas.get((q,), None)
        e1 = oneq.get((q,), None) if oneq else None
        rows.append(
            dict(
                qubit=q,
                t1=t1 if t1 else np.nan,
                t2=t2 if t2 else np.nan,
                frequency=freq if freq else np.nan,
                readout_error=(m.error if m is not None and m.error is not None else np.nan),
                readout_duration=(m.duration if m is not None and m.duration else np.nan),
                err_1q=(e1.error if e1 is not None and e1.error is not None else np.nan),
                dur_1q=(e1.duration if e1 is not None and e1.duration else np.nan),
            )
        )
    qubits = pd.DataFrame(rows).set_index("qubit")
    qubits["one_q_gate"] = oneq_name or "sx"

    # A snapshot with no single-qubit gate error at all is incomplete. We drop it
    # rather than impute a value from other devices -- that would be fabricated data.
    if qubits["err_1q"].isna().all() or qubits["t1"].isna().all():
        return None

    # T2 <= 2*T1 physically; snapshots occasionally violate it -> clamp.
    bad_t2 = qubits["t2"] > 2 * qubits["t1"]
    qubits.loc[bad_t2, "t2"] = 2 * qubits.loc[bad_t2, "t1"]

    # ---- per-coupling -------------------------------------------------------
    props = target[two_q]
    erows = []
    n_pairs = 0
    for pair, inst in props.items():
        if pair is None or len(pair) != 2:
            continue
        n_pairs += 1
        err = inst.error
        if err is None or not math.isfinite(err) or err >= MAX_USABLE_2Q_ERROR:
            continue
        erows.append(
            dict(
                control=pair[0],
                target=pair[1],
                err_2q=float(err),
                dur_2q=float(inst.duration) if inst.duration else np.nan,
            )
        )
    if not erows:
        return None
    edges = pd.DataFrame(erows)
    if n_pairs and len(edges) / n_pairs < MIN_USABLE_EDGE_FRACTION:
        return None  # device is largely uncalibrated (e.g. a snapshot taken mid-failure)

    # Undirected chip graph; keep the better direction's error on the edge.
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for r in edges.itertuples():
        u, v = int(r.control), int(r.target)
        prev = G.get_edge_data(u, v)
        if prev is None or r.err_2q < prev["err_2q"]:
            G.add_edge(u, v, err_2q=float(r.err_2q), dur_2q=float(r.dur_2q))

    # Median-fill missing per-qubit values (rare; a handful of qubits per device).
    for c in ["t1", "t2", "readout_error", "err_1q", "dur_1q", "readout_duration"]:
        qubits[c] = qubits[c].fillna(qubits[c].median())
    qubits["frequency"] = qubits["frequency"].fillna(qubits["frequency"].median())
    if qubits["frequency"].isna().all():
        qubits["frequency"] = 0.0

    return Calibration(
        name=name,
        n_qubits=n,
        two_q_gate=two_q,
        qubits=qubits,
        edges=edges,
        graph=G,
        source=source,
    )


# ---------------------------------------------------------------------------
# Source 1: recorded real-hardware snapshots
# ---------------------------------------------------------------------------

def available_snapshots() -> list[str]:
    from qiskit_ibm_runtime import fake_provider as fp

    return sorted(
        n
        for n in dir(fp)
        if n.startswith("Fake") and "Provider" not in n and "Fractional" not in n
    )


def load_snapshot(cls_name: str) -> Optional[Calibration]:
    """Load one recorded IBM device calibration, e.g. 'FakeSherbrooke'."""
    from qiskit_ibm_runtime import fake_provider as fp

    backend = getattr(fp, cls_name)()
    return calibration_from_target(backend.target, backend.name, source="snapshot")


def load_all_snapshots(min_qubits: int = 5, verbose: bool = True) -> dict[str, Calibration]:
    out: dict[str, Calibration] = {}
    dropped = []
    for cls_name in available_snapshots():
        if cls_name in NON_REPRESENTATIVE:
            dropped.append((cls_name, "flagged non-representative by qiskit"))
            continue
        try:
            cal = load_snapshot(cls_name)
        except Exception as exc:  # pragma: no cover
            dropped.append((cls_name, type(exc).__name__))
            continue
        if cal is None or cal.n_qubits < min_qubits:
            dropped.append((cls_name, "unusable/too-small"))
            continue
        out[cal.name] = cal
    if verbose:
        print(f"[devices] loaded {len(out)} real IBM calibration snapshots; dropped {len(dropped)}")
        for d in dropped:
            print(f"          drop {d[0]}: {d[1]}")
    return out


# ---------------------------------------------------------------------------
# Source 2: live calibration from the user's IBM Quantum account
# ---------------------------------------------------------------------------

def load_live(backend_name: str, token: Optional[str] = None, instance: Optional[str] = None,
              channel: str = "ibm_quantum_platform") -> Calibration:
    """
    Fetch the *current* calibration of a real IBM backend.

        cal = load_live("ibm_sherbrooke", token="<YOUR_TOKEN>")

    Requires network access to IBM Quantum and a valid API token. Everything
    downstream treats the result exactly like a snapshot, so you can train on
    snapshots and evaluate against your own live device.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs = {"channel": channel}
    if token:
        kwargs["token"] = token
    if instance:
        kwargs["instance"] = instance
    service = QiskitRuntimeService(**kwargs)
    backend = service.backend(backend_name)
    cal = calibration_from_target(backend.target, backend.name, source="live")
    if cal is None:
        raise RuntimeError(f"{backend_name}: calibration incomplete or unusable")
    return cal


def live_snapshot_series(backend_name: str, out_dir: str, token: Optional[str] = None) -> str:
    """
    Append today's live calibration to an on-disk time series. Run this on a
    schedule (e.g. daily) to accumulate genuine temporal drift data for the
    device you actually use, then point the dataset builder at the directory.
    """
    import json
    import os
    from datetime import datetime, timezone

    os.makedirs(out_dir, exist_ok=True)
    cal = load_live(backend_name, token=token)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"{cal.name}_{stamp}.json")
    payload = {
        "name": cal.name,
        "n_qubits": cal.n_qubits,
        "two_q_gate": cal.two_q_gate,
        "captured_utc": stamp,
        "qubits": cal.qubits.reset_index().to_dict(orient="records"),
        "edges": cal.edges.to_dict(orient="records"),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh)
    return path


def load_saved_snapshot(path: str) -> Calibration:
    """Rehydrate a `Calibration` saved by `live_snapshot_series`."""
    import json

    with open(path) as fh:
        p = json.load(fh)
    qubits = pd.DataFrame(p["qubits"]).set_index("qubit")
    edges = pd.DataFrame(p["edges"])
    G = nx.Graph()
    G.add_nodes_from(range(p["n_qubits"]))
    for r in edges.itertuples():
        u, v = int(r.control), int(r.target)
        prev = G.get_edge_data(u, v)
        if prev is None or r.err_2q < prev["err_2q"]:
            G.add_edge(u, v, err_2q=float(r.err_2q), dur_2q=float(r.dur_2q))
    return Calibration(
        name=p["name"],
        n_qubits=p["n_qubits"],
        two_q_gate=p["two_q_gate"],
        qubits=qubits,
        edges=edges,
        graph=G,
        source=f"live@{p['captured_utc']}",
    )
