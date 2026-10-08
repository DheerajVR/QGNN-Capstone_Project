#!/usr/bin/env python3
"""
Full experimental protocol:

  * three splits   -- random / held-out device / held-out circuit family
  * four methods   -- mean, calibration budget, gradient-boosted trees, MLP, GNN
  * ablations      -- feature groups, message-passing depth, structure

Everything reported in the write-up is produced by this script and written to
results/results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from qgnn.baselines import run_baselines                      # noqa: E402
from qgnn.data import (EDGE_GROUP_HARDWARE, EDGE_GROUP_WORKLOAD, GROUP_HARDWARE,  # noqa: E402
                       GROUP_PRIOR, GROUP_WORKLOAD, split_groups, split_indices)
from qgnn.dataset import load_dataset                          # noqa: E402
from qgnn.metrics import (graph_metrics, hotspot_metrics,       # noqa: E402
                          node_regression_metrics, random_baseline_hotspot)
from qgnn.train import build_split, predict, train_model        # noqa: E402


def evaluate(name, node_reg, node_cls, graph_reg, truth, gid, raw_infid):
    return {
        "method": name,
        "node_regression": node_regression_metrics(node_reg, truth["y_node_reg"], gid),
        "hotspot": hotspot_metrics(node_cls, truth["y_node_cls"], gid, raw_infid),
        "graph": graph_metrics(graph_reg, truth["y_graph"]),
    }


def run_split(samples, mode, seed, epochs, hidden, layers, verbose=True):
    print(f"\n=== split: {mode} ===", flush=True)
    tr_it, va_it, te_it, norm, (tr, va, te) = build_split(samples, mode, seed=seed)
    info = {
        "mode": mode,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "test_devices": split_groups(samples, te, "device")[:60],
        "test_families": split_groups(samples, te, "family"),
    }
    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}")
    if mode == "device":
        print(f"  held-out devices: {info['test_devices']}")
    if mode == "family":
        print(f"  held-out families: {info['test_families']}")

    results = []

    # ---- baselines ---------------------------------------------------------
    t0 = time.time()
    base, btruth = run_baselines(samples, tr, te, seed=seed)
    for name in ["mean", "budget", "gbt"]:
        results.append(evaluate(
            {"mean": "Train mean", "budget": "Calibration budget",
             "gbt": "Gradient-boosted trees"}[name],
            base[name]["node_reg"], base[name]["node_cls"], base[name]["graph_reg"],
            btruth, btruth["graph_id"], btruth["raw_infid"]))
    rb = random_baseline_hotspot(btruth["y_node_cls"], btruth["graph_id"])
    results.append({"method": "Random ranking", "node_regression": None,
                    "hotspot": rb, "graph": None})
    print(f"  baselines done in {time.time()-t0:.0f}s")

    # ---- learned models ----------------------------------------------------
    for kind, label in [("mlp", "MLP (no graph structure)"), ("gnn", "QChipGNN")]:
        print(f"  training {label} ...", flush=True)
        # the MLP is ~10x cheaper per epoch and converges more slowly; give it
        # the epochs it needs so the comparison is not a budget artefact
        ep = epochs * 2 if kind == "mlp" else epochs
        model, hist = train_model(tr_it, va_it, kind=kind, epochs=ep,
                                  hidden=hidden, layers=layers, seed=seed, verbose=verbose)
        pred, truth, gid = predict(model, te_it)
        r = evaluate(label, pred["node_reg"], pred["node_cls"], pred["graph_reg"],
                     truth, gid, truth["raw_infid"])
        r["epochs_run"] = len(hist)
        r["n_params"] = int(sum(p.numel() for p in model.parameters()))
        results.append(r)
        if kind == "mlp":
            np.savez(f"results/preds_mlp_{mode}.npz", node_reg=pred["node_reg"],
                     node_cls=pred["node_cls"], graph_reg=pred["graph_reg"], gid=gid)
        if kind == "gnn":
            info["gnn_history"] = hist
            torch.save({"state": model.state_dict(), "norm": norm.state(),
                        "hidden": hidden, "layers": layers, "dropout": 0.1},
                       f"results/model_{mode}.pt")
            np.savez(f"results/preds_{mode}.npz",
                     node_reg=pred["node_reg"], node_cls=pred["node_cls"],
                     graph_reg=pred["graph_reg"], gid=gid,
                     y_node_reg=truth["y_node_reg"], y_node_cls=truth["y_node_cls"],
                     y_graph=truth["y_graph"], raw_infid=truth["raw_infid"],
                     budget=truth["budget_total"], test_idx=np.asarray(te))
            np.savez(f"results/preds_baseline_{mode}.npz",
                     budget_node=base["budget"]["node_reg"],
                     budget_cls=base["budget"]["node_cls"],
                     budget_graph=base["budget"]["graph_reg"],
                     gbt_node=base["gbt"]["node_reg"],
                     gbt_cls=base["gbt"]["node_cls"],
                     gbt_graph=base["gbt"]["graph_reg"],
                     y_node_reg=btruth["y_node_reg"], y_node_cls=btruth["y_node_cls"],
                     y_graph=btruth["y_graph"], gid=btruth["graph_id"])
    info["results"] = results
    return info


def run_ablations(samples, seed, epochs, hidden, layers, full_seeds=3):
    """
    Held-out-device split. Feature groups are removed cleanly: the analytic
    error-budget features mix calibration and workload, so any ablation that
    claims to remove one of those must remove the budget group too, or the
    information leaks straight back in.
    """
    print("\n=== ablations (held-out-device split) ===", flush=True)
    HW, WL, PR = GROUP_HARDWARE, GROUP_WORKLOAD, GROUP_PRIOR
    configs = [
        ("full", None, None, layers, full_seeds),
        ("calibration only (no workload at all)", WL + PR, EDGE_GROUP_WORKLOAD, layers, 1),
        ("workload only (no calibration at all)", HW + PR, EDGE_GROUP_HARDWARE, layers, 1),
        ("no analytic error-budget features", PR, None, layers, 1),
        ("no raw calibration (budget kept)", HW, EDGE_GROUP_HARDWARE, layers, 1),
        ("no raw workload counts (budget kept)", WL, EDGE_GROUP_WORKLOAD, layers, 1),
        ("1 message-passing layer", None, None, 1, 1),
        ("2 message-passing layers", None, None, 2, 1),
        ("6 message-passing layers", None, None, 6, 1),
    ]
    out = []
    for name, dn, de, L, n_seeds in configs:
        print(f"  -- {name}", flush=True)
        runs = []
        for s in range(n_seeds):
            tr_it, va_it, te_it, _, _ = build_split(
                samples, "device", seed=seed, drop_node_cols=dn, drop_edge_cols=de)
            model, hist = train_model(tr_it, va_it, kind="gnn", epochs=epochs,
                                      hidden=hidden, layers=L, seed=seed + 101 * s,
                                      verbose=False)
            pred, truth, gid = predict(model, te_it)
            r = evaluate(name, pred["node_reg"], pred["node_cls"], pred["graph_reg"],
                         truth, gid, truth["raw_infid"])
            r["epochs_run"] = len(hist)
            r["train_seed"] = seed + 101 * s
            runs.append(r)
            print(f"     seed {r['train_seed']}: "
                  f"MAE={r['node_regression']['mae_log10']:.3f} "
                  f"rho_within={r['node_regression']['spearman_within_circuit']:.3f} "
                  f"prec@m={r['hotspot']['precision_at_m']:.3f} "
                  f"graph_rho={r['graph']['spearman']:.3f}", flush=True)
        main = runs[0]
        if len(runs) > 1:
            main = dict(main)
            main["seed_spread"] = {
                "mae_log10": [r["node_regression"]["mae_log10"] for r in runs],
                "spearman_within_circuit": [
                    r["node_regression"]["spearman_within_circuit"] for r in runs],
                "precision_at_m": [r["hotspot"]["precision_at_m"] for r in runs],
                "graph_spearman": [r["graph"]["spearman"] for r in runs],
            }
        out.append(main)
    return out


def dataset_summary(samples):
    fam, dev = {}, {}
    ks, fids, infids = [], [], []
    for s in samples:
        fam[s["family"]] = fam.get(s["family"], 0) + 1
        dev[s["device"]] = dev.get(s["device"], 0) + 1
        ks.append(int(s["x"].shape[0]))
        fids.append(float(s["y_graph_fid"]))
        infids.extend(s["y_node_infid"].tolist())
    infids = np.array(infids)
    return {
        "n_samples": len(samples),
        "n_devices": len(dev),
        "n_families": len(fam),
        "n_nodes": int(sum(ks)),
        "k_distribution": {int(k): int((np.array(ks) == k).sum()) for k in sorted(set(ks))},
        "per_family": fam,
        "circuit_fidelity": {
            "min": float(np.min(fids)), "p05": float(np.percentile(fids, 5)),
            "median": float(np.median(fids)), "p95": float(np.percentile(fids, 95)),
            "max": float(np.max(fids)),
        },
        "qubit_infidelity": {
            "min": float(infids.min()), "median": float(np.median(infids)),
            "p90": float(np.percentile(infids, 90)),
            "p99": float(np.percentile(infids, 99)), "max": float(infids.max()),
            "frac_above_1pct": float((infids > 0.01).mean()),
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.pkl.gz")
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--ablation-epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablations", action="store_true",
                    help="run the legacy inline ablations; the maintained set lives in 02b")
    ap.add_argument("--family-seeds", type=int, default=1,
                    help="repeat the held-out-family split with N different draws")
    a = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    samples = load_dataset(a.data)
    print(f"[exp] {len(samples)} samples loaded")
    out = {"dataset": dataset_summary(samples), "config": vars(a), "splits": {}}
    print(json.dumps(out["dataset"]["circuit_fidelity"], indent=2))

    for mode in ["random", "device", "family"]:
        out["splits"][mode] = run_split(samples, mode, a.seed, a.epochs, a.hidden, a.layers)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2, default=float)

    # The family split has only 12 groups, so one draw of held-out families is a
    # weak estimate. Repeat it with different draws and report the spread.
    if a.family_seeds > 1:
        reps = []
        for s in range(1, a.family_seeds):
            r = run_split(samples, "family", a.seed + 17 * s, a.epochs, a.hidden,
                          a.layers, verbose=False)
            r.pop("gnn_history", None)
            reps.append(r)
            with open(a.out, "w") as fh:
                json.dump({**out, "family_repeats": reps}, fh, indent=2, default=float)
        out["family_repeats"] = reps

    if a.ablations:
        out["ablations"] = run_ablations(samples, a.seed, a.ablation_epochs, a.hidden, a.layers)

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n[exp] wrote {a.out}")
