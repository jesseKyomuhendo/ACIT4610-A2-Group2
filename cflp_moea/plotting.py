"""Plots: final Pareto fronts (f1 vs f2) of both MOEAs on the same axes, etc.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cflp_moea.algorithms.common import unique_non_dominated  # noqa: E402

# same colour and a different marker per MOEA, so the plots also work in black and white
STYLE = {
    "nsga2": dict(label="NSGA-II", color="#2a78d6", marker="o"),
    "spea2": dict(label="SPEA2", color="#eb6834", marker="x"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def combined_front(fronts: list[np.ndarray]) -> np.ndarray:
    """Non-dominated points of all runs together, sorted by f1."""
    F = np.vstack(fronts)
    _, F = unique_non_dominated(F, F)
    return F[np.argsort(F[:, 0])]


def _style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))


def plot_instance(fronts: dict, instance: str, configs: list[str], path) -> None:
    """One panel per configuration; each panel shows both MOEAs' fronts (all runs combined).

    fronts[(config, algorithm)] = list of final fronts, one per run.
    """
    fig, axes = plt.subplots(1, len(configs), figsize=(4.2 * len(configs), 3.8),
                             sharex=True, sharey=True)
    for ax, cfg in zip(np.atleast_1d(axes), configs):
        for alg, st in STYLE.items():
            F = combined_front(fronts[(cfg, alg)])
            if st["marker"] == "o":   # hollow circles so SPEA2's crosses show through
                ax.scatter(F[:, 0], F[:, 1], s=46, marker="o", facecolors="none",
                           edgecolors=st["color"], linewidths=1.4, label=st["label"], zorder=3)
            else:
                ax.scatter(F[:, 0], F[:, 1], s=34, marker="x", color=st["color"],
                           linewidths=1.4, label=st["label"], zorder=4)
        ax.set_title(f"Configuration {cfg}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("f1: facility-opening cost", color=INK, fontsize=9)
        _style_axes(ax)
    np.atleast_1d(axes)[0].set_ylabel("f2: customer-allocation cost", color=INK, fontsize=9)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(f"{instance}: final Pareto fronts (10 runs combined)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_hv_boxplots(df, instance: str, configs: list[str], path) -> None:
    """HV over the runs per configuration, the two MOEAs side by side."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    positions, data, colors = [], [], []
    for k, cfg in enumerate(configs):
        for o, (alg, st) in enumerate(STYLE.items()):
            hv = df[(df.instance == instance) & (df.config == cfg) & (df.algorithm == alg)]["hv"]
            positions.append(k * 3 + o)
            data.append(hv.to_numpy())
            colors.append(st["color"])
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, alpha=0.35, edgecolor=c)
    ax.set_xticks([k * 3 + 0.5 for k in range(len(configs))], [f"Config {c}" for c in configs])
    ax.set_ylabel("Hypervolume (normalised)", color=INK, fontsize=9)
    ax.set_title(f"{instance}: hypervolume over 10 runs", color=INK, fontsize=10, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=st["color"], alpha=0.35,
                             edgecolor=st["color"]) for st in STYLE.values()]
    ax.legend(handles, [st["label"] for st in STYLE.values()], frameon=False, fontsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
