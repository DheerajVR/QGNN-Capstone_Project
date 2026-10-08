"""
Chip-as-graph featurisation.

A sample is one (device, qubit-patch, circuit) triple:

  nodes  = the physical qubits the circuit runs on
  edges  = the physical couplings between them (whether or not the circuit uses them)

Node features fuse two things the model must reason about jointly:
  * what the hardware is  -- T1, T2, readout error, 1q error, incident 2q errors
  * what the circuit asks of it -- gate counts, busy time, idle time, schedule span

Circuit *identity* is never a feature. The model only ever sees physics and
workload structure, so held-out-family evaluation is meaningful.
"""

from __future__ import annotations

import numpy as np

NODE_FEATURES = [
    # --- hardware ---
    "log_t1", "log_t2", "t2_over_t1", "log_readout_err", "log_err_1q",
    "log_dur_1q", "log_readout_dur", "device_degree", "patch_degree",
    "log_mean_e2q", "log_min_e2q", "log_max_e2q", "log_mean_d2q",
    # --- workload ---
    "n_1q", "n_2q", "log_busy", "log_idle", "log_span",
    "idle_over_t1", "idle_over_t2", "busy_over_t1", "idle_fraction",
    "used_edge_count",
    # --- fused physics priors ---
    "acc_1q_err", "acc_2q_err", "log_budget_gates", "log_budget_total",
    "log_decoh", "budget_rank_in_patch",
]

EDGE_FEATURES = [
    "log_err_2q", "log_dur_2q", "n_uses", "is_used", "acc_edge_err",
    "log_edge_active_time", "err_2q_rank_in_patch",
]

GLOBAL_FEATURES = [
    "k", "log_device_size", "log_total_duration", "log_depth",
    "log_n_2q_total", "log_n_1q_total", "twoq_per_qubit", "log_patch_edges",
]

_EPS = 1e-12


def _log10(x, floor=1e-12):
    return np.log10(np.maximum(np.asarray(x, dtype=float), floor))


def build_sample_graph(cal, patch, sub_target, tqc, sched, stats, gt, meta):
    """Turn one simulated (device, patch, circuit) triple into arrays."""
    k = len(patch)
    loc = {p: i for i, p in enumerate(patch)}
    q = cal.qubits
    G = cal.graph

    # ---- patch edges (undirected, induced subgraph) --------------------------
    pe = []
    for a in range(k):
        for b in range(a + 1, k):
            u, v = patch[a], patch[b]
            if G.has_edge(u, v):
                d = G[u][v]
                pe.append((a, b, float(d["err_2q"]), float(d["dur_2q"])))
    if not pe:
        return None

    # ---- how the circuit actually uses each qubit / coupling -----------------
    edge_uses = {(a, b): 0 for a, b, _, _ in pe}
    edge_time = {(a, b): 0.0 for a, b, _, _ in pe}
    twoq_name = cal.two_q_gate
    for inst in tqc.data:
        if inst.operation.name != twoq_name:
            continue
        i, j = sorted(tqc.find_bit(x).index for x in inst.qubits)
        if (i, j) in edge_uses:
            edge_uses[(i, j)] += 1
            try:
                edge_time[(i, j)] += sub_target[twoq_name][(i, j)].duration or 0.0
            except Exception:
                pass

    # per-qubit accumulated two-qubit error from the couplings it actually drove
    acc_2q = np.zeros(k)
    used_deg = np.zeros(k)
    for (a, b), n in edge_uses.items():
        if n == 0:
            continue
        e = next(x[2] for x in pe if x[0] == a and x[1] == b)
        acc_2q[a] += n * e
        acc_2q[b] += n * e
        used_deg[a] += 1
        used_deg[b] += 1

    t1 = q.loc[patch, "t1"].to_numpy(float)
    t2 = q.loc[patch, "t2"].to_numpy(float)
    ro = q.loc[patch, "readout_error"].to_numpy(float)
    e1 = q.loc[patch, "err_1q"].to_numpy(float)
    d1 = q.loc[patch, "dur_1q"].to_numpy(float)
    rod = q.loc[patch, "readout_duration"].to_numpy(float)

    n1 = stats["n_1q"].astype(float)
    n2 = stats["n_2q"].astype(float)
    busy = stats["busy"].astype(float)
    idle = stats["idle"].astype(float)
    span = stats["span"].astype(float)
    total = float(stats["total_duration"])

    inc = [[x[2] for x in pe if a in (x[0], x[1])] for a in range(k)]
    incd = [[x[3] for x in pe if a in (x[0], x[1])] for a in range(k)]
    mean_e2 = np.array([np.mean(v) if v else np.nan for v in inc])
    min_e2 = np.array([np.min(v) if v else np.nan for v in inc])
    max_e2 = np.array([np.max(v) if v else np.nan for v in inc])
    mean_d2 = np.array([np.mean(v) if v else np.nan for v in incd])
    med = np.nanmedian(mean_e2) if np.isfinite(mean_e2).any() else 1e-3
    mean_e2, min_e2, max_e2 = (np.nan_to_num(x, nan=med) for x in (mean_e2, min_e2, max_e2))
    mean_d2 = np.nan_to_num(mean_d2, nan=float(np.nanmedian(mean_d2)) if np.isfinite(mean_d2).any() else 3e-7)

    acc_1q = n1 * e1
    # Analytic error budget -- the physics estimate a calibration-only heuristic makes.
    surv_gates = np.exp(-acc_1q) * np.exp(-acc_2q)
    decoh = 1.0 - np.exp(-np.divide(idle, np.maximum(t2, _EPS)))
    budget_gates = 1.0 - surv_gates
    budget_total = 1.0 - surv_gates * (1.0 - decoh) * (1.0 - ro)
    rank = np.argsort(np.argsort(budget_total)) / max(1, k - 1)

    device_deg = np.array([G.degree(p) for p in patch], dtype=float)
    patch_deg = np.array([sum(1 for x in pe if a in (x[0], x[1])) for a in range(k)], dtype=float)

    X = np.stack([
        _log10(t1), _log10(t2), t2 / np.maximum(t1, _EPS), _log10(ro), _log10(e1),
        _log10(d1), _log10(rod), device_deg, patch_deg,
        _log10(mean_e2), _log10(min_e2), _log10(max_e2), _log10(mean_d2),
        n1, n2, _log10(busy), _log10(idle), _log10(span),
        idle / np.maximum(t1, _EPS), idle / np.maximum(t2, _EPS),
        busy / np.maximum(t1, _EPS), idle / max(total, _EPS), used_deg,
        acc_1q, acc_2q, _log10(budget_gates), _log10(budget_total),
        _log10(decoh), rank,
    ], axis=1).astype(np.float32)
    assert X.shape[1] == len(NODE_FEATURES), (X.shape, len(NODE_FEATURES))

    e_err = np.array([x[2] for x in pe])
    e_dur = np.array([x[3] for x in pe])
    e_use = np.array([edge_uses[(x[0], x[1])] for x in pe], dtype=float)
    e_time = np.array([edge_time[(x[0], x[1])] for x in pe], dtype=float)
    e_rank = np.argsort(np.argsort(e_err)) / max(1, len(pe) - 1)
    E = np.stack([
        _log10(e_err), _log10(e_dur), e_use, (e_use > 0).astype(float),
        e_use * e_err, _log10(e_time), e_rank,
    ], axis=1).astype(np.float32)
    assert E.shape[1] == len(EDGE_FEATURES)

    src = np.array([x[0] for x in pe], dtype=np.int64)
    dst = np.array([x[1] for x in pe], dtype=np.int64)
    # store undirected once; the model symmetrises at message-passing time

    U = np.array([
        k, np.log10(cal.n_qubits), _log10(total), np.log10(max(tqc.depth(), 1)),
        np.log10(max(n2.sum() / 2, 1)), np.log10(max(n1.sum(), 1)),
        n2.sum() / (2 * k), np.log10(max(len(pe), 1)),
    ], dtype=np.float32)
    assert U.shape[0] == len(GLOBAL_FEATURES)

    return dict(
        x=X, edge_attr=E, src=src, dst=dst, u=U,
        y_node_infid=gt.q_infidelity.astype(np.float32),
        y_node_dev=gt.q_marginal_dev.astype(np.float32),
        y_graph_fid=np.float32(gt.circuit_fidelity),
        y_graph_tvd=np.float32(gt.circuit_tvd),
        budget_total=budget_total.astype(np.float32),
        budget_gates=budget_gates.astype(np.float32),
        patch=np.array(patch, dtype=np.int64),
        device=meta["device"],
        family=meta["family"],
        depth_param=meta["depth_param"],
        bias=meta["bias"],
        n_qubits_device=cal.n_qubits,
        family_group=meta["family"],
    )


def hotspot_labels(y: np.ndarray, frac: float = 1 / 3) -> np.ndarray:
    """Top-`frac` of qubits within this circuit, by true error. At least one."""
    k = len(y)
    m = max(1, int(np.ceil(frac * k)))
    idx = np.argsort(-y)[:m]
    out = np.zeros(k, dtype=np.float32)
    out[idx] = 1.0
    return out
