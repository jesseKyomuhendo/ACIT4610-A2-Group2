"""Compute HV, #non-dominated, runtime tables + statistical tests -> results/tables/.

Usage:
    python scripts/analyze_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cflp_moea.metrics import REF_POINT, bounds, hypervolume_2d, n_non_dominated, normalise  # noqa: E402
from cflp_moea.stats import compare, summary  # noqa: E402
from cflp_moea.utils import load_results  # noqa: E402


def per_run_table(results: list[dict]) -> tuple[pd.DataFrame, dict]:
    """One row per run with HV (shared scaling per instance), #non-dominated and runtime."""
    rows, scaling = [], {}
    for inst in sorted({r["instance"] for r in results}):
        runs = [r for r in results if r["instance"] == inst]
        # same bounds for both MOEAs: min/max over every final front of this instance
        lo, hi = bounds([np.array(r["F"]) for r in runs])
        scaling[inst] = {"f1_min": lo[0], "f1_max": hi[0], "f2_min": lo[1], "f2_max": hi[1],
                         "reference_point": REF_POINT.tolist()}
        for r in runs:
            F = np.array(r["F"])
            rows.append({"instance": inst, "config": r["config"], "algorithm": r["algorithm"],
                         "run": r["run"], "seed": r["seed"],
                         "hv": hypervolume_2d(normalise(F, lo, hi)),
                         "n_nd": n_non_dominated(F), "time_s": r["time_s"],
                         "n_evals": r["n_evals"], "best_f1_plus_f2": F.sum(axis=1).min()})
    return pd.DataFrame(rows), scaling


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (inst, cfg, alg), g in df.groupby(["instance", "config", "algorithm"]):
        hv = summary(g["hv"])
        rows.append({"instance": inst, "config": cfg, "algorithm": alg, "runs": len(g),
                     "hv_mean": hv["mean"], "hv_std": hv["std"], "hv_best": hv["best"],
                     "hv_worst": hv["worst"], "n_nd_mean": g["n_nd"].mean(),
                     "n_nd_std": g["n_nd"].std(ddof=1), "time_mean_s": g["time_s"].mean()})
    return pd.DataFrame(rows)


def test_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (inst, cfg), g in df.groupby(["instance", "config"]):
        a = g[g.algorithm == "nsga2"].sort_values("run")["hv"]
        b = g[g.algorithm == "spea2"].sort_values("run")["hv"]
        res = compare(a, b)
        winner = {"first": "NSGA-II", "second": "SPEA2"}.get(res["winner"], res["winner"])
        rows.append({"instance": inst, "config": cfg, "p_value": res["p_value"],
                     "a12_nsga2_vs_spea2": res["a12"], "result": winner})
    return pd.DataFrame(rows)


def main() -> None:
    results = load_results(ROOT / "results/raw")
    if not results:
        sys.exit("no results in results/raw - run scripts/run_experiments.py first")
    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)

    runs, scaling = per_run_table(results)
    summ, tests = summary_table(runs), test_table(runs)

    runs.to_csv(out / "per_run.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    tests.to_csv(out / "tests.csv", index=False)
    (out / "normalisation.json").write_text(json.dumps(scaling, indent=2))
    with open(out / "summary.md", "w") as f:
        f.write("## HV, non-dominated solutions and runtime\n\n")
        f.write(summ.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## Mann-Whitney U test on HV (alpha = 0.05)\n\n")
        f.write(tests.to_markdown(index=False, floatfmt=".4f") + "\n")

    pd.set_option("display.width", 200)
    print(summ.round(4).to_string(index=False))
    print()
    print(tests.round(4).to_string(index=False))
    print(f"\nTables written to {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
