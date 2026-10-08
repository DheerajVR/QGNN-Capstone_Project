#!/usr/bin/env python3
"""Produce every figure in the write-up from results/ artefacts."""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from qgnn.dataset import load_dataset           # noqa: E402
from qgnn.devices import load_snapshot, available_snapshots  # noqa: E402
from qgnn.metrics import _safe_spearman         # noqa: E402
from qgnn import viz                            # noqa: E402

FIG = "figures"


def get(results, mode, method, path):
    for r in results["splits"][mode]["results"]:
        if r["method"] == method:
            cur = r
            for p in path:
                cur = cur[p] if cur is not None else None
            return cur
    return None


def main():
    os.makedirs(FIG, exist_ok=True)
    with open("results/results.json") as fh:
        R = json.load(fh)
    samples = load_dataset("data/dataset.pkl.gz")

    # --- 1. dataset landscape ------------------------------------------------
    viz.fig_dataset(samples, f"{FIG}/01_dataset.png")

    # --- 2. method comparison, held-out devices ------------------------------
    order = ["Train mean", "Random ranking", "Calibration budget",
             "Gradient-boosted trees", "MLP (no graph structure)", "QChipGNN"]
    rows = []
    for m in order:
        nr = get(R, "device", m, ["node_regression"])
        ho = get(R, "device", m, ["hotspot"])
        rows.append((m,
                     (nr or {}).get("spearman_within_circuit", np.nan),
                     (ho or {}).get("precision_at_m", np.nan),
                     (ho or {}).get("exact_worst_qubit_acc", np.nan)))
    viz.fig_method_comparison(rows, f"{FIG}/02_methods_device_split.png",
                              title="Predicting which qubits fail — held-out devices")

    rows_f = []
    for m in order:
        nr = get(R, "family", m, ["node_regression"])
        ho = get(R, "family", m, ["hotspot"])
        rows_f.append((m,
                       (nr or {}).get("spearman_within_circuit", np.nan),
                       (ho or {}).get("precision_at_m", np.nan),
                       (ho or {}).get("exact_worst_qubit_acc", np.nan)))
    viz.fig_method_comparison(rows_f, f"{FIG}/03_methods_family_split.png",
                              title="Predicting which qubits fail — held-out circuit families")

    # --- 3. predicted vs true, node level ------------------------------------
    p = np.load("results/preds_device.npz", allow_pickle=True)
    pb = np.load("results/preds_baseline_device.npz", allow_pickle=True)
    viz.fig_pred_vs_true([
        ("Calibration budget", pb["budget_node"], pb["y_node_reg"]),
        ("Gradient-boosted trees", pb["gbt_node"], pb["y_node_reg"]),
        ("QChipGNN", p["node_reg"], p["y_node_reg"]),
    ], f"{FIG}/04_pred_vs_true_node.png")

    # --- 4. graph-level fidelity --------------------------------------------
    viz.fig_graph_level(p["graph_reg"], p["y_graph"], f"{FIG}/05_circuit_fidelity.png")

    # --- 5. per-family ranking quality ---------------------------------------
    te_idx = p["test_idx"]
    fams = np.array([samples[i]["family"] for i in te_idx])
    gid = p["gid"]
    per_fam = []
    for fam in sorted(set(fams.tolist())):
        gsel = np.where(fams == fam)[0]
        rg, rb, n = [], [], 0
        for g in gsel:
            m = gid == g
            if m.sum() >= 3:
                a = _safe_spearman(p["node_reg"][m], p["y_node_reg"][m])
                b = _safe_spearman(pb["budget_node"][m], pb["y_node_reg"][m])
                if np.isfinite(a):
                    rg.append(a)
                if np.isfinite(b):
                    rb.append(b)
                n += 1
        if rg:
            per_fam.append((fam, float(np.mean(rg)), float(np.mean(rb)) if rb else 0.0, n))
    viz.fig_per_family(per_fam, f"{FIG}/06_per_family.png")
    with open("results/per_family.json", "w") as fh:
        json.dump([{"family": f, "gnn_rho": a, "budget_rho": b, "n_circuits": n}
                   for f, a, b, n in per_fam], fh, indent=2)

    # --- 6. ablations ---------------------------------------------------------
    if "ablations" in R:
        arows = [(a["method"], a["node_regression"]["mae_log10"],
                  a["node_regression"]["spearman_within_circuit"],
                  a["hotspot"]["precision_at_m"]) for a in R["ablations"]]
        viz.fig_ablation(arows, f"{FIG}/07_ablations.png")

    # --- 7. chip heatmaps for two real circuits ------------------------------
    # pick the largest held-out device that actually hosts wide patches
    counts = {}
    for i in te_idx:
        d = samples[i]["device"]
        counts.setdefault(d, [0, samples[i]["n_qubits_device"]])
        counts[d][0] += 1
    best_dev = max(counts, key=lambda d: (counts[d][1], counts[d][0]))
    cls = next(c for c in available_snapshots()
               if (lambda x: x is not None and x.name == best_dev)(_try(c)))
    cal = load_snapshot(cls)

    cand = [g for g in range(len(te_idx))
            if samples[te_idx[g]]["device"] == best_dev
            and (gid == g).sum() >= 6]
    spread = sorted(cand, key=lambda g: -(p["y_node_reg"][gid == g].max()
                                          - p["y_node_reg"][gid == g].min()))
    picked, seen_fam = [], set()
    for g in spread:                       # one example per circuit family
        f = samples[te_idx[g]]["family"]
        if f in seen_fam:
            continue
        seen_fam.add(f)
        picked.append(g)
        if len(picked) == 3:
            break
    examples = []
    for g in picked:
        s = samples[te_idx[g]]
        m = gid == g
        examples.append(dict(
            patch=[int(x) for x in s["patch"]],
            true=p["y_node_reg"][m],
            pred=p["node_reg"][m],
            title=f"{s['family'].replace('_',' ')}, {int(m.sum())} qubits",
        ))
    if examples:
        viz.fig_chip_heatmap(cal, examples, f"{FIG}/08_chip_heatmap.png")

    print("figures written to", FIG)
    for f in sorted(os.listdir(FIG)):
        print("  ", f)


def _try(cls_name):
    try:
        return load_snapshot(cls_name)
    except Exception:
        return None


if __name__ == "__main__":
    main()
