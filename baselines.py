"""
Baselines the GNN has to beat.

1. `mean`      -- predict the training mean. Sanity floor.
2. `budget`    -- the analytic calibration heuristic. For each qubit,
                    eps = 1 - exp(-n1q*e1q) * exp(-sum_used e2q) * exp(-idle/T2) * (1-readout)
                  then a 1-D linear fit from log10(eps) to log10(true infidelity)
                  on the training set. This is what you would do today with
                  nothing but the calibration table -- the honest strong prior.
3. `gbt`       -- gradient-boosted trees on the full node feature vector plus
                  global features. Strong structure-blind tabular learner.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .data import to_targets
from .graph import hotspot_labels


def _flat(samples, idx):
    X, U, gid, y, ycls, bud = [], [], [], [], [], []
    for j, i in enumerate(idx):
        g = samples[i]
        t = to_targets(g)
        n = g["x"].shape[0]
        X.append(g["x"])
        U.append(np.repeat(g["u"][None, :], n, 0))
        gid.append(np.full(n, j))
        y.append(t["y_node_reg"])
        ycls.append(t["y_node_cls"])
        bud.append(g["budget_total"])
    return (np.concatenate(X), np.concatenate(U), np.concatenate(gid),
            np.concatenate(y), np.concatenate(ycls), np.concatenate(bud))


def _graph_level(samples, idx):
    Ug, yg = [], []
    for i in idx:
        g = samples[i]
        t = to_targets(g)
        n = g["x"].shape[0]
        agg = np.concatenate([g["x"].mean(0), g["x"].max(0), g["x"].min(0), g["u"]])
        Ug.append(agg)
        yg.append(t["y_graph"])
    return np.stack(Ug), np.array(yg)


def run_baselines(samples, tr_idx, te_idx, seed=0):
    Xtr, Utr, gtr, ytr, ctr, btr = _flat(samples, tr_idx)
    Xte, Ute, gte, yte, cte, bte = _flat(samples, te_idx)
    out = {}

    # --- 1. mean -------------------------------------------------------------
    out["mean"] = {
        "node_reg": np.full_like(yte, ytr.mean()),
        "node_cls": np.zeros_like(yte),
    }

    # --- 2. analytic calibration budget --------------------------------------
    lb_tr = np.log10(np.clip(btr, 1e-9, 1.0))
    lb_te = np.log10(np.clip(bte, 1e-9, 1.0))
    A = np.stack([lb_tr, np.ones_like(lb_tr)], 1)
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    out["budget"] = {
        "node_reg": coef[0] * lb_te + coef[1],
        "node_cls": lb_te,  # ranking score: higher budget = more likely hotspot
        "fit": {"slope": float(coef[0]), "intercept": float(coef[1])},
    }

    # --- 3. gradient-boosted trees on the same features ----------------------
    Ftr = np.concatenate([Xtr, Utr], 1)
    Fte = np.concatenate([Xte, Ute], 1)
    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.07,
                                        max_depth=None, random_state=seed)
    reg.fit(Ftr, ytr)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.07,
                                         random_state=seed)
    clf.fit(Ftr, ctr)
    out["gbt"] = {
        "node_reg": reg.predict(Fte),
        "node_cls": clf.predict_proba(Fte)[:, 1],
    }

    # --- graph-level counterparts -------------------------------------------
    Gtr, ygtr = _graph_level(samples, tr_idx)
    Gte, ygte = _graph_level(samples, te_idx)
    out["mean"]["graph_reg"] = np.full_like(ygte, ygtr.mean())
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.07, random_state=seed)
    gb.fit(Gtr, ygtr)
    out["gbt"]["graph_reg"] = gb.predict(Gte)

    # budget-based circuit fidelity: product of per-qubit survivals
    def budget_graph(idx):
        vals = []
        for i in idx:
            b = np.clip(samples[i]["budget_total"], 0, 0.999999)
            vals.append(np.log10(np.clip(1 - np.prod(1 - b), 1e-6, 1.0)))
        return np.array(vals)

    gb_tr, gb_te = budget_graph(tr_idx), budget_graph(te_idx)
    A2 = np.stack([gb_tr, np.ones_like(gb_tr)], 1)
    c2, *_ = np.linalg.lstsq(A2, ygtr, rcond=None)
    out["budget"]["graph_reg"] = c2[0] * gb_te + c2[1]

    truth = {"y_node_reg": yte, "y_node_cls": cte, "y_graph": ygte,
             "graph_id": gte, "raw_infid": np.concatenate(
                 [samples[i]["y_node_infid"] for i in te_idx])}
    return out, truth
