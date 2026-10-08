#!/usr/bin/env python3
"""Build the labelled dataset from real IBM calibration snapshots."""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from qgnn.dataset import build_dataset  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dataset.pkl.gz")
    ap.add_argument("--per-device", type=int, default=130)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    build_dataset(a.out, per_device=a.per_device, workers=a.workers, seed=a.seed)
