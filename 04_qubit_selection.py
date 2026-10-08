#!/usr/bin/env python3
"""
The end-use test: does the model actually help you choose qubits?

For each held-out device and each task (circuit family + width) we
  1. draw N candidate qubit patches on the real chip,
  2. simulate every one of them exactly to get the TRUE circuit fidelity,
  3. ask each strategy to pick one patch without seeing those results,
  4. report the true fidelity of what each strategy picked.

Strategies: random, calibration budget (today's practice), QChipGNN, oracle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from qgnn.circuits import sample_circuit                       # noqa: E402
from qgnn.devices import available_snapshots, load_snapshot     # noqa: E402
from qgnn.hardware import (asap_schedule, build_sub_target, ground_truth,  # noqa: E402
                           sample_patch, transpile_to_patch)
from qgnn.predict import featurise, load_model, predict_graph   # noqa: E402

TASKS = [("quantum_volume", 6), ("hardware_efficient_ansatz", 7),
         ("trotter_ising", 7), ("qft", 5), ("qaoa_maxcut", 6), ("ghz", 8)]


def eval_device(cls_name, model_path, n_candidates, seed):
    warnings.filterwarnings("ignore")
    cal = load_snapshot(cls_name)
    if cal is None:
        return []
    model, norm = load_model(model_path)
    rng = np.random.default_rng(seed)
    rows = []

    for family, k in TASKS:
        k = min(k, cal.n_qubits)
        logical, _ = sample_circuit(family, k, rng)
        cands, seen = [], set()
        for _ in range(n_candidates * 4):
            if len(cands) >= n_candidates:
                break
            bias = str(rng.choice(["random", "good", "bad"]))
            patch = sample_patch(cal, k, rng, bias=bias)
            key = tuple(sorted(patch))
            if len(patch) != k or key in seen:
                continue
            seen.add(key)
            tseed = int(rng.integers(1 << 30))
            try:
                sub_target = build_sub_target(cal, patch)
                tqc = transpile_to_patch(logical, sub_target, seed=tseed)
                if tqc is None or tqc.size() == 0 or tqc.size() > 4000:
                    continue
                sched, stats = asap_schedule(tqc, sub_target)
                gt = ground_truth(sched, sub_target)
                if gt is None:
                    continue
                g = featurise(cal, patch, logical, seed=tseed)
                if g is None:
                    continue
                pred = predict_graph(g, model, norm)
            except Exception:
                continue
            cands.append(dict(
                patch=patch,
                true_fid=float(gt.circuit_fidelity),
                true_worst_q=int(patch[int(np.argmax(gt.q_infidelity))]),
                pred_fid=float(pred["circuit_fidelity"]),
                pred_worst_q=int(patch[int(np.argmax(pred["qubit_infidelity"]))]),
                budget_worst=float(np.max(g["budget_total"])),
                budget_prod=float(1 - np.prod(1 - np.clip(g["budget_total"], 0, 0.999999))),
            ))
        if len(cands) < 5:
            continue
        tf = np.array([c["true_fid"] for c in cands])
        pick = {
            "random": int(rng.integers(len(cands))),
            "budget": int(np.argmin([c["budget_prod"] for c in cands])),
            "gnn": int(np.argmax([c["pred_fid"] for c in cands])),
            "oracle": int(np.argmax(tf)),
        }
        rows.append({
            "device": cal.name, "family": family, "k": k, "n_candidates": len(cands),
            "fid_mean": float(tf.mean()), "fid_min": float(tf.min()), "fid_max": float(tf.max()),
            **{f"fid_{s}": float(tf[i]) for s, i in pick.items()},
            **{f"pctile_{s}": float((tf <= tf[i]).mean()) for s, i in pick.items()},
            "worst_qubit_hit_gnn": float(np.mean(
                [c["pred_worst_q"] == c["true_worst_q"] for c in cands])),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/model_device.pt")
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--candidates", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/qubit_selection.json")
    a = ap.parse_args()

    with open(a.results) as fh:
        R = json.load(fh)
    held_out = set(R["splits"]["device"]["test_devices"])
    name2cls = {}
    for c in available_snapshots():
        try:
            cal = load_snapshot(c)
        except Exception:
            continue
        if cal is not None and cal.name in held_out:
            name2cls[cal.name] = c
    print(f"[select] {len(name2cls)} held-out devices: {sorted(name2cls)}")

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(eval_device, cls, a.model, a.candidates, 500 + i): n
                for i, (n, cls) in enumerate(sorted(name2cls.items()))}
        for f in as_completed(futs):
            got = f.result()
            rows.extend(got)
            print(f"  {futs[f]:22s} +{len(got)} tasks", flush=True)

    summary = {}
    if rows:
        for s in ["random", "budget", "gnn", "oracle"]:
            summary[s] = {
                "mean_true_fidelity": float(np.mean([r[f"fid_{s}"] for r in rows])),
                "mean_percentile": float(np.mean([r[f"pctile_{s}"] for r in rows])),
                "frac_picked_top1": float(np.mean(
                    [r[f"fid_{s}"] >= r["fid_max"] - 1e-12 for r in rows])),
            }
        summary["candidate_pool_mean"] = float(np.mean([r["fid_mean"] for r in rows]))
        summary["n_tasks"] = len(rows)
        summary["worst_qubit_hit_rate_gnn"] = float(
            np.mean([r["worst_qubit_hit_gnn"] for r in rows]))

    with open(a.out, "w") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
