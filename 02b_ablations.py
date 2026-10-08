#!/usr/bin/env python3
"""
Feature-group ablations on the held-out-device split, with clean group
boundaries: features that are products of calibration and workload
(idle/T1, n_1q x err_1q, the analytic budget) belong to neither group alone,
so removing "calibration" also removes them.

Writes results/ablations.json and merges the block into results/results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from qgnn.data import (EDGE_GROUP_HARDWARE, EDGE_GROUP_MIXED, EDGE_GROUP_WORKLOAD,  # noqa: E402
                       GROUP_HARDWARE, GROUP_MIXED, GROUP_PRIOR, GROUP_WORKLOAD)
from qgnn.dataset import load_dataset                       # noqa: E402
from qgnn.metrics import (graph_metrics, hotspot_metrics,    # noqa: E402
                          node_regression_metrics)
from qgnn.train import build_split, predict, train_model     # noqa: E402

HW, WL, MX, PR = GROUP_HARDWARE, GROUP_WORKLOAD, GROUP_MIXED, GROUP_PRIOR
EHW, EWL, EMX = EDGE_GROUP_HARDWARE, EDGE_GROUP_WORKLOAD, EDGE_GROUP_MIXED

CONFIGS = [
    ("full", None, None, 4, 3),
    ("no calibration anywhere", HW + MX, EHW + EMX, 4, 1),
    ("no circuit workload anywhere", WL + MX, EWL + EMX, 4, 1),
    ("no calibration x workload interactions", MX, EMX, 4, 1),
    ("no analytic error budget", PR, None, 4, 1),
    ("no raw calibration (interactions kept)", HW, EHW, 4, 1),
    ("no raw workload counts (interactions kept)", WL, EWL, 4, 1),
    ("1 message-passing layer", None, None, 1, 1),
    ("2 message-passing layers", None, None, 2, 1),
    ("6 message-passing layers", None, None, 6, 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.pkl.gz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    samples = load_dataset(a.data)
    out = []
    for name, dn, de, L, n_seeds in CONFIGS:
        print(f"-- {name}", flush=True)
        runs = []
        for s in range(n_seeds):
            tr_it, va_it, te_it, _, _ = build_split(
                samples, "device", seed=a.seed, drop_node_cols=dn, drop_edge_cols=de)
            model, hist = train_model(tr_it, va_it, kind="gnn", epochs=a.epochs,
                                      hidden=a.hidden, layers=L,
                                      seed=a.seed + 101 * s, verbose=False)
            pred, truth, gid = predict(model, te_it)
            r = {
                "method": name,
                "n_dropped_node_features": len(dn or []),
                "n_dropped_edge_features": len(de or []),
                "layers": L,
                "train_seed": a.seed + 101 * s,
                "epochs_run": len(hist),
                "node_regression": node_regression_metrics(
                    pred["node_reg"], truth["y_node_reg"], gid),
                "hotspot": hotspot_metrics(pred["node_cls"], truth["y_node_cls"],
                                           gid, truth["raw_infid"]),
                "graph": graph_metrics(pred["graph_reg"], truth["y_graph"]),
            }
            runs.append(r)
            print(f"   seed {r['train_seed']}: MAE={r['node_regression']['mae_log10']:.3f}"
                  f" rho_within={r['node_regression']['spearman_within_circuit']:.3f}"
                  f" prec@m={r['hotspot']['precision_at_m']:.3f}"
                  f" graph_rho={r['graph']['spearman']:.3f}", flush=True)
        main_r = dict(runs[0])
        if len(runs) > 1:
            main_r["seed_spread"] = {
                "mae_log10": [r["node_regression"]["mae_log10"] for r in runs],
                "spearman_within_circuit": [
                    r["node_regression"]["spearman_within_circuit"] for r in runs],
                "precision_at_m": [r["hotspot"]["precision_at_m"] for r in runs],
                "graph_spearman": [r["graph"]["spearman"] for r in runs],
            }
        out.append(main_r)
        with open("results/ablations.json", "w") as fh:
            json.dump(out, fh, indent=2, default=float)

    with open("results/results.json") as fh:
        R = json.load(fh)
    R["ablations"] = out
    with open("results/results.json", "w") as fh:
        json.dump(R, fh, indent=2, default=float)
    print("merged into results/results.json")


if __name__ == "__main__":
    main()
