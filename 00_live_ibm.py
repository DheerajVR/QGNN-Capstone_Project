#!/usr/bin/env python3
"""
Talk to your own IBM Quantum account.

  # print the current calibration summary of a real backend
  python scripts/00_live_ibm.py --backend ibm_sherbrooke --token $IBM_QUANTUM_TOKEN

  # append today's calibration to a growing time series (run this daily)
  python scripts/00_live_ibm.py --backend ibm_sherbrooke --save data/live/

  # score a real backend with a trained model: which qubits should I avoid?
  python scripts/00_live_ibm.py --backend ibm_sherbrooke --predict \
         --model results/model_device.pt --family quantum_volume --k 7

The sandbox this repo was developed in has no route to IBM Quantum, so the
shipped results use the recorded hardware snapshots instead. This script is the
path to your own live device; the schema is identical either way.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qgnn.devices import live_snapshot_series, load_live  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--token", default=os.environ.get("IBM_QUANTUM_TOKEN"))
    ap.add_argument("--instance", default=os.environ.get("IBM_QUANTUM_INSTANCE"))
    ap.add_argument("--save", default=None, help="directory to append a snapshot to")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--model", default="results/model_device.pt")
    ap.add_argument("--family", default="quantum_volume")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    if a.save:
        path = live_snapshot_series(a.backend, a.save, token=a.token)
        print("saved", path)
        return

    cal = load_live(a.backend, token=a.token, instance=a.instance)
    print(pd.Series(cal.summary()).to_string())

    if not a.predict:
        return

    from qgnn.predict import rank_patches
    out = rank_patches(cal, a.model, family=a.family, k=a.k, top=a.top)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
