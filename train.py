"""Training / inference loop shared by the GNN and the MLP ablation."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .data import (N_EDGE_FEAT, N_GLOBAL_FEAT, N_NODE_FEAT, Normaliser, batches,
                   collate, prepare, split_indices)
from .model import NodeMLP, QChipGNN, multitask_loss

torch.set_num_threads(2)


def make_model(kind: str, hidden=128, layers=4, dropout=0.1):
    if kind == "gnn":
        return QChipGNN(N_NODE_FEAT, N_EDGE_FEAT, N_GLOBAL_FEAT, hidden, layers, dropout)
    if kind == "mlp":
        return NodeMLP(N_NODE_FEAT, N_GLOBAL_FEAT, hidden, dropout)
    raise ValueError(kind)


@torch.no_grad()
def predict(model, items, bs=256):
    model.eval()
    out = {k: [] for k in ["node_reg", "node_cls", "graph_reg"]}
    truth = {k: [] for k in ["y_node_reg", "y_node_cls", "y_graph", "raw_infid", "budget_total"]}
    gid, off = [], 0
    for i in range(0, len(items), bs):
        b = collate(items[i:i + bs])
        p = model(b)
        for k in out:
            out[k].append(p[k].numpy())
        for k in truth:
            truth[k].append(b[k].numpy())
        gid.append(b["node_batch"].numpy() + off)
        off += b["u"].shape[0]
    return ({k: np.concatenate(v) for k, v in out.items()},
            {k: np.concatenate(v) for k, v in truth.items()},
            np.concatenate(gid))


def train_model(train_items, val_items, kind="gnn", epochs=60, bs=64, lr=2e-3,
                hidden=128, layers=4, dropout=0.1, patience=12, seed=0, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(kind, hidden, layers, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * (len(train_items) // bs + 1)),
        pct_start=0.2)

    best, best_state, bad = np.inf, None, 0
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for b in batches(train_items, bs, shuffle=True, rng=rng):
            opt.zero_grad()
            loss, _ = multitask_loss(model(b), b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            try:
                sched.step()
            except ValueError:
                pass
            tot += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vtot, vnb = 0.0, 0
            for b in batches(val_items, 256, shuffle=False):
                vloss, _ = multitask_loss(model(b), b)
                vtot += vloss.item(); vnb += 1
        vl = vtot / max(vnb, 1)
        hist.append({"epoch": ep, "train": tot / max(nb, 1), "val": vl})
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"    ep {ep:3d}  train {tot/max(nb,1):.4f}  val {vl:.4f}"
                  f"  best {best:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        if bad >= patience:
            if verbose:
                print(f"    early stop at epoch {ep}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def build_split(samples, mode, seed=0, drop_node_cols=None, drop_edge_cols=None):
    tr, va, te = split_indices(samples, mode=mode, seed=seed)
    norm = Normaliser().fit(samples, tr)
    items = prepare(samples, norm, drop_node_cols, drop_edge_cols)
    return ([items[i] for i in tr], [items[i] for i in va], [items[i] for i in te],
            norm, (tr, va, te))
