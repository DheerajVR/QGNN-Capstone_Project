"""
Figures. Palette follows the validated categorical/sequential defaults:
slot1 blue #2a78d6, slot2 orange #eb6834, slot3 aqua #1baf7a, slot4 yellow #eda100.
Sequential magnitude uses the single blue ramp, light -> dark.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seqblue", BLUE_RAMP)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "600",
    "figure.dpi": 150, "savefig.bbox": "tight",
    "font.family": ["DejaVu Sans"],
})


def _clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(True, axis=grid_axis, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------

def fig_dataset(samples, path):
    fid = np.array([float(s["y_graph_fid"]) for s in samples])
    inf = np.concatenate([s["y_node_infid"] for s in samples])
    inf = np.clip(inf, 1e-6, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    ax = axes[0]
    ax.hist(fid, bins=45, color=S1, edgecolor=SURFACE, linewidth=0.6)
    ax.set_xlabel("circuit fidelity (Hellinger)")
    ax.set_ylabel("circuits")
    ax.set_title("Circuit outcome spans clean to destroyed")
    _clean(ax)

    ax = axes[1]
    ax.hist(np.log10(inf), bins=45, color=S2, edgecolor=SURFACE, linewidth=0.6)
    ax.set_xlabel("log$_{10}$ per-qubit infidelity")
    ax.set_ylabel("qubits")
    ax.set_title("Per-qubit error spans five orders of magnitude")
    _clean(ax)

    ax = axes[2]
    by_dev = {}
    for s in samples:
        by_dev.setdefault(s["device"], []).append(float(s["y_graph_fid"]))
    names = sorted(by_dev, key=lambda d: np.median(by_dev[d]))
    med = [np.median(by_dev[d]) for d in names]
    ax.plot(range(len(names)), med, "o", color=S3, ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_xlabel(f"real IBM device (n={len(names)}), sorted")
    ax.set_ylabel("median circuit fidelity")
    ax.set_title("Device quality varies widely across the fleet")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_method_comparison(rows, path, title="Held-out devices"):
    """rows: list of (label, within_circuit_rho, precision_at_m, exact_worst_acc)."""
    labels = [r[0] for r in rows]
    metrics = [
        ("Within-circuit rank\ncorrelation (Spearman)", [r[1] for r in rows], S1),
        ("Precision@m\n(hotspot set)", [r[2] for r in rows], S2),
        ("Exact worst-qubit\naccuracy", [r[3] for r in rows], S3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    y = np.arange(len(labels))
    for ax, (name, vals, col) in zip(axes, metrics):
        vals = np.array([np.nan if v is None else v for v in vals], dtype=float)
        bars = ax.barh(y, vals, color=col, height=0.62)
        for b in bars:
            b.set_path_effects([])
        ax.set_yticks(y, labels if ax is axes[0] else [""] * len(labels))
        ax.invert_yaxis()
        ax.set_title(name)
        _clean(ax, grid_axis="x")
        lo = min(0, float(np.nanmin(vals)) * 1.15)
        ax.set_xlim(lo, max(1.0, float(np.nanmax(vals)) * 1.18))
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]), yi,
                        f"{v:.3f}", va="center", fontsize=9, color=INK2)
    fig.suptitle(title, y=1.04, fontsize=12, color=INK, fontweight="600")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_pred_vs_true(pred_sets, path):
    """pred_sets: list of (label, pred_log10, true_log10)."""
    n = len(pred_sets)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.3))
    if n == 1:
        axes = [axes]
    lo, hi = -6.2, 0.3
    for ax, (label, p, t) in zip(axes, pred_sets):
        hb = ax.hexbin(t, p, gridsize=48, extent=(lo, hi, lo, hi),
                       cmap=SEQ, mincnt=1, linewidths=0)
        ax.plot([lo, hi], [lo, hi], "-", color=INK2, lw=1.4, alpha=0.7, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("true log$_{10}$ qubit infidelity")
        if ax is axes[0]:
            ax.set_ylabel("predicted log$_{10}$ qubit infidelity")
        mae = float(np.mean(np.abs(np.asarray(p) - np.asarray(t))))
        ax.set_title(f"{label}   MAE {mae:.3f} dex")
        _clean(ax, grid_axis="both")
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label("qubits", color=INK2)
        cb.outline.set_edgecolor(GRID)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_graph_level(pred, true, path, label="QChipGNN"):
    fid_p = 1 - 10 ** np.asarray(pred)
    fid_t = 1 - 10 ** np.asarray(true)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ax = axes[0]
    ax.hexbin(fid_t, fid_p, gridsize=42, extent=(0, 1, 0, 1), cmap=SEQ,
              mincnt=1, linewidths=0)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1.4, alpha=0.7)
    ax.set_xlabel("true circuit fidelity"); ax.set_ylabel("predicted circuit fidelity")
    ax.set_title(f"{label}: circuit-level fidelity")
    _clean(ax, grid_axis="both")

    ax = axes[1]
    err = fid_p - fid_t
    ax.hist(err, bins=50, color=S1, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(0, color=INK2, lw=1.2)
    ax.set_xlabel("predicted - true fidelity")
    ax.set_ylabel("circuits")
    ax.set_title(f"Residuals: median |err| {np.median(np.abs(err)):.4f}")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_chip_heatmap(cal, examples, path):
    """
    examples: list of dicts with keys patch, true, pred, title.
    Draws the real chip graph, greying the untouched qubits and colouring the
    circuit's qubits by true (top row) and predicted (bottom row) error.
    """
    n = len(examples)
    # zoom: only the neighbourhood around the circuit's qubits, or a 156-qubit
    # chip reduces an 8-qubit patch to a smudge
    ctx = set()
    for ex in examples:
        for q in ex["patch"]:
            ctx.add(q)
            ctx |= set(nx.single_source_shortest_path_length(cal.graph, q, cutoff=3))
    sub_all = cal.graph.subgraph(ctx)
    pos = nx.kamada_kawai_layout(sub_all)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 8.2))
    axes = np.atleast_2d(axes)
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, ex in enumerate(examples):
        vals_all = np.concatenate([ex["true"], ex["pred"]])
        vmin, vmax = float(vals_all.min()), float(vals_all.max())
        if vmax - vmin < 1e-9:
            vmax = vmin + 1e-9
        for i, key in enumerate(["true", "pred"]):
            ax = axes[i, j]
            nx.draw_networkx_edges(sub_all, pos, ax=ax, edge_color="#e8e7e3", width=1.4)
            nx.draw_networkx_nodes(sub_all, pos, ax=ax, node_size=40,
                                   node_color="#efeeea", linewidths=0)
            sub = cal.graph.subgraph(ex["patch"])
            nx.draw_networkx_edges(sub, pos, ax=ax, edge_color=INK2, width=2.0)
            sc = ax.scatter([pos[p][0] for p in ex["patch"]],
                            [pos[p][1] for p in ex["patch"]],
                            c=ex[key], cmap=SEQ, vmin=vmin, vmax=vmax,
                            s=230, edgecolors=SURFACE, linewidths=1.8, zorder=4)
            worst = int(np.argmax(ex[key]))
            ax.scatter([pos[ex["patch"][worst]][0]], [pos[ex["patch"][worst]][1]],
                       s=520, facecolors="none", edgecolors=S2, linewidths=2.4, zorder=5)
            ax.set_title(("measured  " if key == "true" else "predicted  ") + ex["title"],
                         fontsize=10)
            ax.axis("off")
            if i == 1:
                cb = fig.colorbar(sc, ax=axes[:, j].tolist(), fraction=0.045,
                                  pad=0.02, location="bottom", shrink=0.85)
                cb.set_label("log$_{10}$ qubit infidelity", color=INK2, fontsize=8)
                cb.ax.tick_params(labelsize=8)
                cb.outline.set_edgecolor(GRID)
    fig.suptitle(f"{cal.name}  —  orange ring marks the worst qubit "
                 f"(top: measured, bottom: predicted)", y=0.995, fontsize=11,
                 color=INK, fontweight="600")
    fig.savefig(path)
    plt.close(fig)


def fig_ablation(rows, path):
    """rows: list of (name, mae_log10, within_rho, precision_at_m)."""
    names = [r[0] for r in rows]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.55 * len(names) + 2.4))
    for ax, (idx, title, col, lower_better) in zip(axes, [
        (1, "MAE (dex, lower is better)", S4, True),
        (2, "Within-circuit Spearman", S1, False),
        (3, "Precision@m", S3, False),
    ]):
        vals = np.array([r[idx] for r in rows], float)
        ax.barh(y, vals, color=col, height=0.6)
        ax.set_yticks(y, names if ax is axes[0] else [""] * len(names))
        ax.invert_yaxis()
        ax.set_title(title)
        _clean(ax, grid_axis="x")
        ax.set_xlim(0, float(np.nanmax(vals)) * 1.22)
        for yi, v in zip(y, vals):
            ax.text(v * 1.02, yi, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_per_family(rows, path):
    """rows: list of (family, gnn_rho, budget_rho, n)."""
    rows = sorted(rows, key=lambda r: -r[1])
    names = [r[0].replace("_", " ") for r in rows]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(names) + 1.9))
    ax.barh(y - 0.20, [r[1] for r in rows], height=0.36, color=S1, label="QChipGNN")
    ax.barh(y + 0.20, [r[2] for r in rows], height=0.36, color=S4,
            label="Calibration budget")
    span = max(0.05, max(max(r[1], r[2]) for r in rows))
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.012 * span, yi - 0.20, f"{r[1]:.2f}", va="center",
                fontsize=8, color=INK2)
        ax.text(r[2] + 0.012 * span, yi + 0.20, f"{r[2]:.2f}", va="center",
                fontsize=8, color=INK2)
    ax.set_xlim(min(0, min(min(r[1], r[2]) for r in rows) * 1.2), span * 1.16)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("within-circuit Spearman correlation")
    ax.set_title("Per-qubit ranking quality by circuit family (held-out devices)")
    ax.legend(frameon=False, loc="lower right")
    _clean(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
