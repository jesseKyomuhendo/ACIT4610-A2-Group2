"""Pareto-front plots and results tables for the report.

plot_pareto_fronts()  - f1 against f2, NSGA-II and SPEA2 on the same axes
save_results_table()  - prints the table, saves it as CSV and as a PNG image

Both write into FIGURES_DIR / TABLES_DIR from config.yaml.
"""

import csv

import matplotlib

matplotlib.use("Agg")   # draw to files only; works without a screen
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.util.fetch_config import FIGURES_DIR, PLOT_DPI, TABLES_DIR

# One fixed colour and marker per algorithm, so they are told apart by shape
# as well as colour (also readable in black-and-white print).
STYLE = {
    "NSGA-II": {"color": "#2a78d6", "marker": "o"},   # blue, circles
    "SPEA2":   {"color": "#eb6834", "marker": "^"},   # orange, triangles
}
TEXT_COLOR = "#3d3d3a"
GRID_COLOR = "#e4e3dd"

thousands = FuncFormatter(lambda value, _: f"{value:,.0f}")


def plot_pareto_fronts(instance_name, fronts, caption_note=""):
    """Plot the final fronts of both algorithms on the same axes.

    fronts:       {"NSGA-II": array (k, 2), "SPEA2": array (k, 2)}, raw objective values
    caption_note: short text shown under the title, e.g. "C1, combined front of 10 runs"
    Returns the path of the saved PNG.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for algorithm, front in fronts.items():
        front = front[front[:, 0].argsort()]          # sort by f1 so the line follows the front
        style = STYLE[algorithm]
        ax.plot(front[:, 0], front[:, 1], color=style["color"], linewidth=1.5,
                marker=style["marker"], markersize=7, markeredgecolor="white",
                markeredgewidth=1, label=f"{algorithm} ({len(front)} points)")

    fig.suptitle(f"Final Pareto fronts on {instance_name}", color=TEXT_COLOR, fontsize=12,
                 x=0.02, ha="left")
    if caption_note:
        ax.set_title(caption_note, color="#73726c", fontsize=9, loc="left")
    ax.set_xlabel("f1: facility-opening cost (minimise)", color=TEXT_COLOR)
    ax.set_ylabel("f2: customer-allocation cost (minimise)", color=TEXT_COLOR)
    ax.xaxis.set_major_formatter(thousands)
    ax.yaxis.set_major_formatter(thousands)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    path = FIGURES_DIR / f"pareto_{instance_name}.png"
    fig.savefig(path, dpi=PLOT_DPI, facecolor="white")
    plt.close(fig)
    return path


def save_results_table(instance_name, rows):
    """Print the results table and save it as CSV and PNG.

    rows: list of dicts with the same keys, one per algorithm and configuration, e.g.
      {"Config": "C1", "Algorithm": "NSGA-II", "HV mean ± std": "0.8123 ± 0.0040", ...}
    Values should already be formatted as text (analyze_results.py does that).
    Returns (csv_path, png_path).
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())

    # 1. print as aligned text
    widths = [max(len(str(h)), *(len(str(r[h])) for r in rows)) for h in headers]
    print(f"\nResults for {instance_name}")
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(row[h]).ljust(w) for h, w in zip(headers, widths)))

    # 2. CSV
    csv_path = TABLES_DIR / f"results_{instance_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    # 3. PNG image of the table (handy for the report; the report caption names the instance)
    fig, ax = plt.subplots(figsize=(1.35 * len(headers), 0.4 * (len(rows) + 2)))
    ax.axis("off")
    table = ax.table(cellText=[[row[h] for h in headers] for row in rows],
                     colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    table.auto_set_column_width(list(range(len(headers))))   # fit each column to its text
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#b5b4ad")
        if r == 0:
            cell.set_text_props(weight="bold", color=TEXT_COLOR)
            cell.set_facecolor("#f0efea")
    png_path = TABLES_DIR / f"results_{instance_name}.png"
    fig.savefig(png_path, dpi=PLOT_DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return csv_path, png_path