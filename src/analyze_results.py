"""Compute metrics, statistical tests, tables and Pareto plots from the raw results.

Reads RAW_DIR/<instance>.json written by src/run_experiments.py. For each instance:
  1. Normalisation bounds: ideal and nadir over ALL final fronts of the
     instance (both algorithms, all configurations, all runs).
  2. Per run: hypervolume of the normalised front (reference point
     HV_REFERENCE_POINT), number of non-dominated solutions, runtime.
  3. Per configuration: mean, standard deviation (sample, ddof=1), best and
     worst HV; mean #ND; mean time; paired Wilcoxon test NSGA-II vs SPEA2.
  4. Outputs in TABLES_DIR / FIGURES_DIR:
       results_<instance>.csv / .png   the results table for the report
       hv_runs_<instance>.csv          every run's HV, #ND and time
       pareto_<instance>.png           Pareto fronts of both algorithms
An infeasible instance is reported with its reason instead.

Run from the project root (after run_experiments):   python -m src.analyze_results
"""

import csv
import json

import numpy as np

from src.metrics import count_non_dominated, hypervolume_2d, normalization_bounds, normalize
from src.objectives import non_dominated
from src.plotting import plot_pareto_fronts, save_results_table
from src.stats_tests import compare
from src.util.fetch_config import (CONFIG_TO_PLOT, CONFIGURATIONS, FRONT_TO_PLOT,
                                   HV_REFERENCE_POINT, INSTANCES, RAW_DIR, TABLES_DIR)

ALGORITHMS = ["NSGA-II", "SPEA2"]


def load_raw(name):
    path = RAW_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.run_experiments` first.")
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def compute_run_metrics(runs, ideal, nadir):
    """Add hv and n_nd to every run (in place)."""
    for run in runs:
        front = np.array(run["front"])
        run["hv"] = hypervolume_2d(normalize(front, ideal, nadir), HV_REFERENCE_POINT)
        run["n_nd"] = count_non_dominated(front)


def select_runs(runs, algorithm, config):
    """Runs of one algorithm and configuration, ordered by run number (keeps pairs aligned)."""
    chosen = [r for r in runs if r["algorithm"] == algorithm and r["config"] == config]
    return sorted(chosen, key=lambda r: r["run"])


def build_table_rows(runs):
    """One row per configuration and algorithm, with the p-value on the first row."""
    rows = []
    for config in CONFIGURATIONS:
        per_algorithm = {a: select_runs(runs, a, config) for a in ALGORITHMS}
        hv = {a: np.array([r["hv"] for r in per_algorithm[a]]) for a in ALGORITHMS}
        test = compare(hv["NSGA-II"], hv["SPEA2"])

        for position, algorithm in enumerate(ALGORITHMS):
            n_nd = np.array([r["n_nd"] for r in per_algorithm[algorithm]])
            times = np.array([r["time_s"] for r in per_algorithm[algorithm]])
            values = hv[algorithm]
            rows.append({
                "Config": config,
                "Algorithm": algorithm,
                "HV mean ± std": f"{values.mean():.4f} ± {values.std(ddof=1):.4f}",
                "HV best": f"{values.max():.4f}",
                "HV worst": f"{values.min():.4f}",
                "#ND mean ± std": f"{n_nd.mean():.1f} ± {n_nd.std(ddof=1):.1f}",
                "Time (s)": f"{times.mean():.2f}",
                "p (HV)": f"{test['p_value']:.4f}" if position == 0 else "",
                # "n.s." = not significant at level ALPHA
                "Better": (test["better"] if test["better"] in ALGORITHMS else "n.s.") if position == 0 else "",
            })
    return rows


def front_for_plot(runs, algorithm):
    """The front to plot for one algorithm, as chosen by FRONT_TO_PLOT in config.yaml."""
    chosen = select_runs(runs, algorithm, CONFIG_TO_PLOT)
    if FRONT_TO_PLOT == "best_run":
        return np.array(max(chosen, key=lambda r: r["hv"])["front"])
    all_points = np.vstack([r["front"] for r in chosen])     # "combined"
    return non_dominated(all_points)


def save_run_metrics(name, runs):
    path = TABLES_DIR / f"hv_runs_{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["config", "algorithm", "run", "seed", "hv", "n_nd", "time_s"])
        for r in sorted(runs, key=lambda r: (r["config"], r["algorithm"], r["run"])):
            writer.writerow([r["config"], r["algorithm"], r["run"], r["seed"],
                             f"{r['hv']:.6f}", r["n_nd"], f"{r['time_s']:.3f}"])


def analyze_instance(category, name):
    raw = load_raw(name)
    print(f"\n{'=' * 70}\n[{category}] {name}")
    if raw["status"] == "infeasible":
        print(f"  INFEASIBLE: {raw['reason']}")
        return

    runs = raw["runs"]
    ideal, nadir = normalization_bounds([np.array(r["front"]) for r in runs])
    print(f"  normalisation: ideal = [{ideal[0]:,.2f}, {ideal[1]:,.2f}], "
          f"nadir = [{nadir[0]:,.2f}, {nadir[1]:,.2f}], reference point = {HV_REFERENCE_POINT}")

    compute_run_metrics(runs, ideal, nadir)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    save_run_metrics(name, runs)
    save_results_table(name, build_table_rows(runs))

    fronts = {a: front_for_plot(runs, a) for a in ALGORITHMS}
    note = (f"{CONFIG_TO_PLOT}, combined non-dominated set of all runs" if FRONT_TO_PLOT == "combined"
            else f"{CONFIG_TO_PLOT}, run with the highest hypervolume")
    print(f"  plot saved: {plot_pareto_fronts(name, fronts, note)}")


def main():
    for category, name in INSTANCES.items():
        analyze_instance(category, name)


if __name__ == "__main__":
    main()