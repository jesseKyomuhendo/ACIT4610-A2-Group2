"""Produce Pareto-front plots (one per small/medium/large instance) -> results/figures/.

Usage:
    python scripts/make_plots.py        (run analyze_results.py first for the HV boxplots)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cflp_moea.plotting import plot_hv_boxplots, plot_instance  # noqa: E402
from cflp_moea.utils import load_results  # noqa: E402


def main() -> None:
    results = load_results(ROOT / "results/raw")
    if not results:
        sys.exit("no results in results/raw - run scripts/run_experiments.py first")
    out = ROOT / "results/figures"
    out.mkdir(parents=True, exist_ok=True)

    fronts = defaultdict(lambda: defaultdict(list))
    for r in results:
        fronts[r["instance"]][(r["config"], r["algorithm"])].append(np.array(r["F"]))
    configs = sorted({r["config"] for r in results})

    per_run = ROOT / "results/tables/per_run.csv"
    df = pd.read_csv(per_run) if per_run.exists() else None

    for inst in sorted(fronts):
        plot_instance(fronts[inst], inst, configs, out / f"pareto_{inst}.png")
        print(f"saved results/figures/pareto_{inst}.png")
        if df is not None:
            plot_hv_boxplots(df, inst, configs, out / f"hv_{inst}.png")
            print(f"saved results/figures/hv_{inst}.png")


if __name__ == "__main__":
    main()
