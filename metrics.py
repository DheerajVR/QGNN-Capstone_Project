"""Evaluation metrics. Everything reported in the paper comes from here."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_spearman(a, b):
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else np.nan


def node_regression_metrics(pred_log, true_log, graph_id):
    """pred_log / true_log are log10 per-qubit infidelity, concatenated over graphs."""
    pred_log = np.asarray(pred_log, float)
    true_log = np.asarray(true_log, float)
    lin_p, lin_t = 10 ** pred_log, 10 ** true_log

    per_circ = []
    for g in np.unique(graph_id):
        m = graph_id == g
        if m.sum() >= 3:
            r = _safe_spearman(pred_log[m], true_log[m])
            if np.isfinite(r):
                per_circ.append(r)

    return {
        "mae_log10": float(np.mean(np.abs(pred_log - true_log))),
        "rmse_log10": float(np.sqrt(np.mean((pred_log - true_log) ** 2))),
        "r2_log10": float(1 - np.sum((pred_log - true_log) ** 2) /
                          max(np.sum((true_log - true_log.mean()) ** 2), 1e-12)),
        "spearman_global": _safe_spearman(pred_log, true_log),
        "spearman_within_circuit": float(np.mean(per_circ)) if per_circ else float("nan"),
        "mae_linear": float(np.mean(np.abs(lin_p - lin_t))),
        "median_ae_linear": float(np.median(np.abs(lin_p - lin_t))),
        "n_nodes": int(len(pred_log)),
        "n_circuits_ranked": len(per_circ),
    }


def hotspot_metrics(score, label, graph_id, true_err=None):
    score = np.asarray(score, float)
    label = np.asarray(label, float)
    out = {}
    if 0 < label.sum() < len(label):
        out["roc_auc"] = float(roc_auc_score(label, score))
        out["pr_auc"] = float(average_precision_score(label, score))
        out["pr_auc_baseline"] = float(label.mean())
    top1, prec_at_m, top1_in_topset = [], [], []
    for g in np.unique(graph_id):
        m = graph_id == g
        s, l = score[m], label[m]
        k = len(s)
        n_pos = int(l.sum())
        if k < 3 or n_pos == 0:
            continue
        order = np.argsort(-s)
        top1_in_topset.append(float(l[order[0]] == 1))
        prec_at_m.append(float(l[order[:n_pos]].sum() / n_pos))
        if true_err is not None:
            te = np.asarray(true_err, float)[m]
            top1.append(float(order[0] == int(np.argmax(te))))
    out["top1_is_hotspot"] = float(np.mean(top1_in_topset)) if top1_in_topset else float("nan")
    out["precision_at_m"] = float(np.mean(prec_at_m)) if prec_at_m else float("nan")
    if top1:
        out["exact_worst_qubit_acc"] = float(np.mean(top1))
    out["n_circuits"] = len(prec_at_m)
    return out


def graph_metrics(pred_log, true_log):
    """log10 circuit infidelity (1 - Hellinger fidelity)."""
    pred_log = np.asarray(pred_log, float)
    true_log = np.asarray(true_log, float)
    # A fidelity is a probability: any deployed model clips to [0, 1], so the
    # reported fidelity error is measured on clipped predictions.
    fid_p = np.clip(1 - 10 ** pred_log, 0.0, 1.0)
    fid_t = np.clip(1 - 10 ** true_log, 0.0, 1.0)
    return {
        "mae_log10": float(np.mean(np.abs(pred_log - true_log))),
        "r2_log10": float(1 - np.sum((pred_log - true_log) ** 2) /
                          max(np.sum((true_log - true_log.mean()) ** 2), 1e-12)),
        "spearman": _safe_spearman(pred_log, true_log),
        "fidelity_mae": float(np.mean(np.abs(fid_p - fid_t))),
        "fidelity_median_ae": float(np.median(np.abs(fid_p - fid_t))),
        "n_circuits": int(len(pred_log)),
    }


def random_baseline_hotspot(label, graph_id, rng_seed=0):
    """Expected precision@m for a random ranker, for reference."""
    rng = np.random.default_rng(rng_seed)
    return hotspot_metrics(rng.random(len(label)), label, graph_id)
