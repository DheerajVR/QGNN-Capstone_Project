"""
Using a trained model: given a chip and a circuit, which qubits will fail?

Two entry points:

  predict_patch(cal, model, family/circuit, patch)
      -> per-qubit predicted infidelity + predicted circuit fidelity

  rank_patches(cal, model_path, family, k)
      -> the qubit patches on this chip ranked best-to-worst for this circuit,
         i.e. a drop-in replacement for "pick the qubits with the lowest
         calibration numbers".

`rank_patches` is the practical payoff: it turns per-qubit error prediction into
a layout decision, and `scripts/04_qubit_selection.py` measures whether taking
its advice actually raises measured circuit fidelity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .circuits import sample_circuit
from .data import Normaliser
from .graph import build_sample_graph
from .hardware import asap_schedule, build_sub_target, sample_patch, transpile_to_patch
from .model import QChipGNN
from .data import N_EDGE_FEAT, N_GLOBAL_FEAT, N_NODE_FEAT


class _NullGT:
    """build_sample_graph wants labels; at inference time there are none."""

    def __init__(self, k):
        self.q_infidelity = np.zeros(k)
        self.q_marginal_dev = np.zeros(k)
        self.circuit_fidelity = 1.0
        self.circuit_tvd = 0.0


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = QChipGNN(N_NODE_FEAT, N_EDGE_FEAT, N_GLOBAL_FEAT,
                     ck.get("hidden", 128), ck.get("layers", 4),
                     dropout=ck.get("dropout", 0.1))
    model.load_state_dict(ck["state"])
    model.eval()
    return model, Normaliser.from_state(ck["norm"])


def featurise(cal, patch, logical, seed=0, meta=None):
    sub_target = build_sub_target(cal, patch)
    tqc = transpile_to_patch(logical, sub_target, seed=seed)
    if tqc is None or tqc.size() == 0:
        return None
    sched, stats = asap_schedule(tqc, sub_target)
    meta = meta or dict(device=cal.name, family="unknown", depth_param=0, bias="n/a")
    return build_sample_graph(cal, patch, sub_target, tqc, sched, stats,
                              _NullGT(len(patch)), meta)


@torch.no_grad()
def predict_graph(g, model, norm):
    x, e, u = norm.apply(g)
    b = dict(
        x=torch.from_numpy(x), edge_attr=torch.from_numpy(e),
        u=torch.from_numpy(u)[None, :],
        src=torch.from_numpy(g["src"]), dst=torch.from_numpy(g["dst"]),
        node_batch=torch.zeros(x.shape[0], dtype=torch.long),
        edge_batch=torch.zeros(g["src"].shape[0], dtype=torch.long),
    )
    p = model(b)
    return dict(
        qubit_infidelity=10 ** p["node_reg"].numpy(),
        hotspot_score=torch.sigmoid(p["node_cls"]).numpy(),
        circuit_fidelity=float(1 - 10 ** p["graph_reg"].item()),
    )


def predict_patch(cal, model, norm, logical, patch, seed=0):
    g = featurise(cal, patch, logical, seed=seed)
    if g is None:
        return None
    out = predict_graph(g, model, norm)
    out["patch"] = list(patch)
    out["budget"] = g["budget_total"]
    return out


def rank_patches(cal, model_path, family="quantum_volume", k=7, n_candidates=250,
                 top=15, seed=0):
    model, norm = load_model(model_path)
    rng = np.random.default_rng(seed)
    logical, _ = sample_circuit(family, k, rng)

    seen, rows = set(), []
    for _ in range(n_candidates * 3):
        if len(rows) >= n_candidates:
            break
        bias = str(rng.choice(["random", "good", "bad"]))
        patch = sample_patch(cal, k, rng, bias=bias)
        key = tuple(sorted(patch))
        if len(patch) != k or key in seen:
            continue
        seen.add(key)
        try:
            r = predict_patch(cal, model, norm, logical, patch, seed=int(rng.integers(1 << 30)))
        except Exception:
            continue
        if r is None:
            continue
        rows.append({
            "qubits": " ".join(str(q) for q in r["patch"]),
            "pred_circuit_fidelity": r["circuit_fidelity"],
            "pred_worst_qubit": r["patch"][int(np.argmax(r["qubit_infidelity"]))],
            "pred_worst_infidelity": float(np.max(r["qubit_infidelity"])),
            "pred_mean_infidelity": float(np.mean(r["qubit_infidelity"])),
            "calibration_budget_worst": float(np.max(r["budget"])),
        })
    df = pd.DataFrame(rows).sort_values("pred_circuit_fidelity", ascending=False)
    return df.head(top).reset_index(drop=True)
