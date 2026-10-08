"""Splits, target transforms, feature normalisation and batching."""

from __future__ import annotations

import numpy as np
import torch

from .graph import EDGE_FEATURES, GLOBAL_FEATURES, NODE_FEATURES, hotspot_labels

FLOOR = 1e-6


def to_targets(g: dict) -> dict:
    y_inf = np.clip(g["y_node_infid"].astype(np.float64), FLOOR, 1.0)
    y_dev = np.clip(g["y_node_dev"].astype(np.float64), FLOOR, 1.0)
    y_g = np.clip(1.0 - float(g["y_graph_fid"]), FLOOR, 1.0)
    return dict(
        y_node_reg=np.log10(y_inf).astype(np.float32),
        y_node_dev=np.log10(y_dev).astype(np.float32),
        y_node_cls=hotspot_labels(g["y_node_infid"]),
        y_graph=np.float32(np.log10(y_g)),
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def split_indices(samples, mode="random", seed=0, val_frac=0.12, test_frac=0.18):
    rng = np.random.default_rng(seed)
    n = len(samples)
    if mode == "random":
        idx = rng.permutation(n)
        n_test = int(test_frac * n)
        n_val = int(val_frac * n)
        return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]

    key = "device" if mode == "device" else "family"
    groups = sorted({s[key] for s in samples})
    rng.shuffle(groups)
    # With few groups (12 circuit families) a 1-group validation set makes early
    # stopping a coin flip, so enforce a floor of 3 test / 2 validation groups.
    floor_test, floor_val = (3, 2) if mode == "family" else (1, 1)
    n_test = max(floor_test, int(round(test_frac * len(groups))))
    n_val = max(floor_val, int(round(val_frac * len(groups))))
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test + n_val])
    tr, va, te = [], [], []
    for i, s in enumerate(samples):
        (te if s[key] in test_g else va if s[key] in val_g else tr).append(i)
    return np.array(tr), np.array(va), np.array(te)


def split_groups(samples, idx, key):
    return sorted({samples[i][key] for i in idx})


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class Normaliser:
    def __init__(self, clip=8.0):
        self.clip = clip

    def fit(self, samples, train_idx):
        X = np.concatenate([samples[i]["x"] for i in train_idx], 0).astype(np.float64)
        E = np.concatenate([samples[i]["edge_attr"] for i in train_idx], 0).astype(np.float64)
        U = np.stack([samples[i]["u"] for i in train_idx], 0).astype(np.float64)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-8
        self.em, self.es = E.mean(0), E.std(0) + 1e-8
        self.um, self.us = U.mean(0), U.std(0) + 1e-8
        return self

    def _z(self, a, m, s):
        return np.clip((a - m) / s, -self.clip, self.clip).astype(np.float32)

    def apply(self, g):
        return (self._z(g["x"], self.xm, self.xs),
                self._z(g["edge_attr"], self.em, self.es),
                self._z(g["u"], self.um, self.us))

    def state(self):
        return {k: np.asarray(getattr(self, k)).tolist() for k in
                ["xm", "xs", "em", "es", "um", "us"]}

    @classmethod
    def from_state(cls, st, clip=8.0):
        o = cls(clip)
        for k, v in st.items():
            setattr(o, k, np.asarray(v, dtype=np.float64))
        return o


# ---------------------------------------------------------------------------
# Tensors + batching
# ---------------------------------------------------------------------------

# Feature groups, used for ablations. Indices into NODE_FEATURES / EDGE_FEATURES.
#
# The three-way split matters: several features are *products* of calibration
# and workload (idle/T1, n_1q x err_1q, the error budget). An ablation that
# claims to remove calibration but keeps idle/T1 has not removed calibration.
GROUP_HARDWARE = list(range(0, 13))            # pure calibration
GROUP_WORKLOAD = [13, 14, 15, 16, 17, 21, 22]  # pure circuit workload
GROUP_MIXED = [18, 19, 20, 23, 24, 25, 26, 27, 28]  # calibration x workload
GROUP_PRIOR = list(range(23, 29))              # the analytic error budget alone

EDGE_GROUP_HARDWARE = [0, 1, 6]                # err_2q, dur_2q, err rank
EDGE_GROUP_WORKLOAD = [2, 3]                   # n_uses, is_used
EDGE_GROUP_MIXED = [4, 5]                      # n_uses x err_2q, active time


def zero_columns(a, cols):
    if not cols:
        return a
    a = a.copy()
    a[:, cols] = 0.0
    return a


def prepare(samples, norm: Normaliser, drop_node_cols=None, drop_edge_cols=None):
    out = []
    for g in samples:
        x, e, u = norm.apply(g)
        x = zero_columns(x, drop_node_cols)
        e = zero_columns(e, drop_edge_cols)
        t = to_targets(g)
        out.append(dict(
            x=torch.from_numpy(x),
            edge_attr=torch.from_numpy(e),
            u=torch.from_numpy(u),
            src=torch.from_numpy(g["src"]),
            dst=torch.from_numpy(g["dst"]),
            y_node_reg=torch.from_numpy(t["y_node_reg"]),
            y_node_dev=torch.from_numpy(t["y_node_dev"]),
            y_node_cls=torch.from_numpy(t["y_node_cls"]),
            y_graph=torch.tensor(t["y_graph"]),
            budget_total=torch.from_numpy(g["budget_total"]),
            raw_infid=torch.from_numpy(g["y_node_infid"]),
            raw_fid=torch.tensor(np.float32(g["y_graph_fid"])),
            device=g["device"], family=g["family"], k=int(g["x"].shape[0]),
        ))
    return out


def collate(items):
    xs, es, us, srcs, dsts = [], [], [], [], []
    ynr, ynd, ync, yg, bud, raw_i = [], [], [], [], [], []
    nb, eb = [], []
    off = 0
    for i, it in enumerate(items):
        n = it["x"].shape[0]
        xs.append(it["x"]); es.append(it["edge_attr"]); us.append(it["u"])
        srcs.append(it["src"] + off); dsts.append(it["dst"] + off)
        ynr.append(it["y_node_reg"]); ynd.append(it["y_node_dev"])
        ync.append(it["y_node_cls"]); yg.append(it["y_graph"])
        bud.append(it["budget_total"]); raw_i.append(it["raw_infid"])
        nb.append(torch.full((n,), i, dtype=torch.long))
        eb.append(torch.full((it["src"].shape[0],), i, dtype=torch.long))
        off += n
    return dict(
        x=torch.cat(xs), edge_attr=torch.cat(es), u=torch.stack(us),
        src=torch.cat(srcs), dst=torch.cat(dsts),
        node_batch=torch.cat(nb), edge_batch=torch.cat(eb),
        y_node_reg=torch.cat(ynr), y_node_dev=torch.cat(ynd),
        y_node_cls=torch.cat(ync), y_graph=torch.stack(yg),
        budget_total=torch.cat(bud), raw_infid=torch.cat(raw_i),
        graph_sizes=torch.tensor([it["k"] for it in items]),
    )


def batches(items, bs, shuffle=True, rng=None):
    idx = np.arange(len(items))
    if shuffle:
        (rng or np.random).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield collate([items[j] for j in idx[i:i + bs]])


N_NODE_FEAT = len(NODE_FEATURES)
N_EDGE_FEAT = len(EDGE_FEATURES)
N_GLOBAL_FEAT = len(GLOBAL_FEATURES)
