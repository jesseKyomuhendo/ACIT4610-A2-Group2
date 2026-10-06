"""Runs NSGA-II and SPEA2 on every instance, configuration and run, and saves the raw results.

Run it as a module from the project root. Metrics are computed afterwards by analyze_results.
"""

import json
import time

import numpy as np

from src.data_loader import InfeasibleInstanceError, get_instance, verify_data_integrity
from src.nsga2 import run_nsga2
from src.objectives import non_dominated
from src.spea2 import run_spea2
from src.util.fetch_config import BASE_SEED, CONFIGURATIONS, INSTANCES, N_RUNS, RAW_DIR

ALGORITHMS = {"NSGA-II": run_nsga2, "SPEA2": run_spea2}


def run_instance(category, name):
    """Runs all configurations and runs on one instance and returns the record to save."""
    record = {"instance": name, "category": category}
    # An instance without any feasible solution is skipped and the reason is saved instead
    try:
        inst = get_instance(name)
    except InfeasibleInstanceError as error:
        print(f"\n[{category}] {name}: SKIPPED - {error}")
        record.update(status="infeasible", reason=str(error), runs=[])
        return record

    print(f"\n[{category}] {name}: {inst.m} facilities, {inst.n} customers")
    record.update(status="ok", runs=[])

    for config_name, params in CONFIGURATIONS.items():
        for run in range(N_RUNS):
            seed = BASE_SEED + run
            # Both algorithms get a fresh generator with the same seed, so run k forms a fair pair
            for algorithm, run_algorithm in ALGORITHMS.items():
                rng = np.random.default_rng(seed)
                start = time.perf_counter()
                result = run_algorithm(inst, params, rng)
                seconds = time.perf_counter() - start
                front = non_dominated(result["front"])     # distinct points sorted by f1

                record["runs"].append({
                    "algorithm": algorithm,
                    "config": config_name,
                    "run": run,
                    "seed": seed,
                    "time_s": seconds,
                    "evaluations": result["evaluations"],
                    "generations": result["generations"],
                    "front": front.tolist(),                 # one [f1, f2] pair per point
                })
                print(f"  {config_name} run {run + 1:>2}/{N_RUNS}  {algorithm:<7}  "
                      f"{seconds:5.1f} s  {len(front):>3} distinct front points")
    return record


def main():
    """Verifies the data, runs every instance and saves one JSON file per instance."""
    verify_data_integrity()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    # No metrics here, because hypervolume normalisation needs the fronts of all runs first
    for category, name in INSTANCES.items():
        record = run_instance(category, name)
        path = RAW_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as file:
            json.dump(record, file)
        print(f"  saved {path}")

    print(f"\nAll experiments finished in {(time.perf_counter() - total_start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()