"""
QChipGNN -- an edge-conditioned message-passing network over the qubit graph.

Deliberately written in plain PyTorch (no torch-geometric) so the message
passing is visible and the repo has one dependency fewer.

Heads:
  node regression      -> log10 per-qubit infidelity when this circuit runs
  node classification  -> is this qubit an error hotspot for this circuit
  node regression (aux)-> log10 per-qubit measured-marginal deviation
  graph regression     -> log10 circuit infidelity (1 - Hellinger fidelity)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, out_act=False, dropout=0.0):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def scatter_sum(src, index, n):
    out = src.new_zeros((n, src.shape[-1]))
    return out.index_add_(0, index, src)


def scatter_mean(src, index, n):
    s = scatter_sum(src, index, n)
    cnt = src.new_zeros(n).index_add_(0, index, src.new_ones(src.shape[0]))
    return s / cnt.clamp(min=1).unsqueeze(-1)


def scatter_max(src, index, n):
    out = src.new_zeros((n, src.shape[-1]))
    return out.index_reduce_(0, index, src, "amax", include_self=False)


class MPLayer(nn.Module):
    """One round of edge-conditioned message passing with a global context vector."""

    def __init__(self, h, dropout=0.0):
        super().__init__()
        self.msg = mlp([4 * h, 2 * h, h], out_act=True, dropout=dropout)
        self.upd = mlp([3 * h, 2 * h, h], dropout=dropout)
        self.edge_upd = mlp([3 * h, 2 * h, h], dropout=dropout)
        self.norm_n = nn.LayerNorm(h)
        self.norm_e = nn.LayerNorm(h)

    def forward(self, h_n, h_e, src, dst, u, node_batch, edge_batch):
        # symmetrise: every undirected coupling sends a message both ways
        s = torch.cat([src, dst])
        d = torch.cat([dst, src])
        e2 = torch.cat([h_e, h_e], 0)
        ub = u[torch.cat([edge_batch, edge_batch])]
        m = self.msg(torch.cat([h_n[s], h_n[d], e2, ub], -1))
        agg = torch.cat([
            scatter_sum(m, d, h_n.shape[0]),
            scatter_max(m, d, h_n.shape[0]),
        ], -1)
        h_n = self.norm_n(h_n + self.upd(torch.cat([h_n, agg], -1)))
        h_e = self.norm_e(h_e + self.edge_upd(torch.cat([h_e, h_n[src], h_n[dst]], -1)))
        return h_n, h_e


class QChipGNN(nn.Module):
    def __init__(self, n_node_feat, n_edge_feat, n_global_feat,
                 hidden=128, layers=4, dropout=0.1):
        super().__init__()
        h = hidden
        self.enc_n = mlp([n_node_feat, h, h], out_act=True)
        self.enc_e = mlp([n_edge_feat, h, h], out_act=True)
        self.enc_u = mlp([n_global_feat, h, h], out_act=True)
        self.layers = nn.ModuleList([MPLayer(h, dropout) for _ in range(layers)])
        self.u_upd = mlp([3 * h, 2 * h, h])
        self.norm_u = nn.LayerNorm(h)

        self.head_node_reg = mlp([2 * h, h, h // 2, 1])
        self.head_node_cls = mlp([2 * h, h, h // 2, 1])
        self.head_node_dev = mlp([2 * h, h, h // 2, 1])
        self.head_graph = mlp([3 * h, h, h // 2, 1])

    def forward(self, b):
        h_n = self.enc_n(b["x"])
        h_e = self.enc_e(b["edge_attr"])
        u = self.enc_u(b["u"])
        nb, eb = b["node_batch"], b["edge_batch"]
        n_g = u.shape[0]

        for layer in self.layers:
            h_n, h_e = layer(h_n, h_e, b["src"], b["dst"], u, nb, eb)
            pooled = torch.cat([scatter_mean(h_n, nb, n_g), scatter_max(h_n, nb, n_g)], -1)
            u = self.norm_u(u + self.u_upd(torch.cat([u, pooled], -1)))

        h_cat = torch.cat([h_n, u[nb]], -1)
        g_cat = torch.cat([u, scatter_mean(h_n, nb, n_g), scatter_max(h_n, nb, n_g)], -1)
        return {
            "node_reg": self.head_node_reg(h_cat).squeeze(-1),
            "node_cls": self.head_node_cls(h_cat).squeeze(-1),
            "node_dev": self.head_node_dev(h_cat).squeeze(-1),
            "graph_reg": self.head_graph(g_cat).squeeze(-1),
        }


class NodeMLP(nn.Module):
    """Structure-blind baseline: same features, same capacity, no message passing."""

    def __init__(self, n_node_feat, n_global_feat, hidden=128, dropout=0.1):
        super().__init__()
        h = hidden
        self.enc = mlp([n_node_feat + n_global_feat, h, h, h], out_act=True, dropout=dropout)
        self.head_node_reg = mlp([h, h, h // 2, 1])
        self.head_node_cls = mlp([h, h, h // 2, 1])
        self.head_node_dev = mlp([h, h, h // 2, 1])
        self.head_graph = mlp([2 * h, h, h // 2, 1])

    def forward(self, b):
        z = self.enc(torch.cat([b["x"], b["u"][b["node_batch"]]], -1))
        n_g = b["u"].shape[0]
        g = torch.cat([scatter_mean(z, b["node_batch"], n_g),
                       scatter_max(z, b["node_batch"], n_g)], -1)
        return {
            "node_reg": self.head_node_reg(z).squeeze(-1),
            "node_cls": self.head_node_cls(z).squeeze(-1),
            "node_dev": self.head_node_dev(z).squeeze(-1),
            "graph_reg": self.head_graph(g).squeeze(-1),
        }


def multitask_loss(pred, b, w=(1.0, 0.5, 0.3, 1.0)):
    l_reg = F.huber_loss(pred["node_reg"], b["y_node_reg"], delta=1.0)
    l_cls = F.binary_cross_entropy_with_logits(pred["node_cls"], b["y_node_cls"])
    l_dev = F.huber_loss(pred["node_dev"], b["y_node_dev"], delta=1.0)
    l_g = F.huber_loss(pred["graph_reg"], b["y_graph"], delta=1.0)
    total = w[0] * l_reg + w[1] * l_cls + w[2] * l_dev + w[3] * l_g
    return total, dict(node_reg=l_reg.item(), node_cls=l_cls.item(),
                       node_dev=l_dev.item(), graph=l_g.item())
