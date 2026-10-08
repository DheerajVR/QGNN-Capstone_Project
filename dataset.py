"""
Dataset construction: real calibration x circuit workload -> labelled graphs.

Each worker owns one device, loads its real calibration once, and produces
`per_device` samples by repeatedly (a) picking a connected qubit patch,
(b) picking a circuit, (c) transpiling + ASAP-scheduling it onto that patch,
(d) simulating it exactly under the device's own error channels.
"""

from __future__ import annotations

import gzip
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .circuits import FAMILIES, sample_circuit
from .devices import load_all_snapshots, load_snapshot, available_snapshots
from .graph import build_sample_graph
from .hardware import asap_schedule, build_sub_target, ground_truth, sample_patch, transpile_to_patch

# Simulation budget guards -- keep the exact density-matrix simulation tractable.
MAX_TRANSPILED_SIZE = 4000
K_CHOICES = np.array([4, 5, 6, 7, 8, 9])
K_WEIGHTS = np.array([0.14, 0.20, 0.22, 0.20, 0.16, 0.08])
BIASES = ["random", "random", "good", "bad"]


def build_device_samples(cls_name: str, n_samples: int, seed: int) -> list[dict]:
    import warnings

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    cal = load_snapshot(cls_name)
    if cal is None:
        return []

    out: list[dict] = []
    attempts = 0
    max_attempts = n_samples * 6
    while len(out) < n_samples and attempts < max_attempts:
        attempts += 1
        k = int(rng.choice(K_CHOICES, p=K_WEIGHTS))
        k = min(k, cal.n_qubits)
        bias = str(rng.choice(BIASES))
        patch = sample_patch(cal, k, rng, bias=bias)
        if len(patch) != k:
            continue
        try:
            sub_target = build_sub_target(cal, patch)
        except Exception:
            continue
        family = str(rng.choice(FAMILIES))
        try:
            logical, depth_param = sample_circuit(family, k, rng)
        except Exception:
            continue
        tqc = transpile_to_patch(logical, sub_target, seed=int(rng.integers(1 << 30)))
        if tqc is None or tqc.size() == 0 or tqc.size() > MAX_TRANSPILED_SIZE:
            continue
        sched, stats = asap_schedule(tqc, sub_target)
        gt = ground_truth(sched, sub_target)
        if gt is None or not np.isfinite(gt.circuit_fidelity):
            continue
        meta = dict(device=cal.name, family=family, depth_param=depth_param, bias=bias)
        g = build_sample_graph(cal, patch, sub_target, tqc, sched, stats, gt, meta)
        if g is not None:
            out.append(g)
    return out


def build_dataset(out_path: str, per_device: int = 120, workers: int = 2,
                  seed: int = 0, devices: list[str] | None = None) -> dict:
    cals = load_all_snapshots(verbose=True)
    usable = set(cals.keys())
    cls_names = []
    for cls_name in available_snapshots():
        try:
            c = load_snapshot(cls_name)
        except Exception:
            continue
        if c is not None and c.name in usable:
            cls_names.append((cls_name, c.name))
    if devices:
        cls_names = [c for c in cls_names if c[1] in set(devices)]

    print(f"[dataset] {len(cls_names)} devices x {per_device} samples, {workers} workers")
    samples: list[dict] = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(build_device_samples, cn, per_device, seed + 1000 * i): dn
            for i, (cn, dn) in enumerate(cls_names)
        }
        for fut in as_completed(futs):
            dn = futs[fut]
            try:
                got = fut.result()
            except Exception as exc:
                print(f"  [warn] {dn}: {type(exc).__name__}: {exc}")
                got = []
            samples.extend(got)
            done += 1
            el = time.time() - t0
            print(f"  [{done}/{len(cls_names)}] {dn:22s} +{len(got):4d}  "
                  f"total={len(samples):6d}  {el/60:5.1f} min", flush=True)

    with gzip.open(out_path, "wb") as fh:
        pickle.dump(samples, fh, protocol=4)
    print(f"[dataset] wrote {len(samples)} samples -> {out_path} in {(time.time()-t0)/60:.1f} min")
    return {"n_samples": len(samples), "n_devices": len(cls_names)}


def load_dataset(path: str) -> list[dict]:
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)
