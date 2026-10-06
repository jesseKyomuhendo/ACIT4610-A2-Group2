"""Run every (algorithm x configuration x instance x seed) combination and save raw results to results/raw/.

Usage:
    python scripts/run_experiments.py                       # full experiment from configs/
    python scripts/run_experiments.py --runs 2 --jobs 4     # quick check, 4 processes
Runs that already have a result file are skipped, so the script can be restarted.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cflp_moea.algorithms import nsga2, spea2  # noqa: E402
from cflp_moea.data_loader import load_instance  # noqa: E402
from cflp_moea.utils import Timer, make_rng, run_seed, save_json  # noqa: E402

ALGORITHMS = {"nsga2": nsga2, "spea2": spea2}


def run_one(task: dict) -> str:
    inst = load_instance(task["instance"])
    seed = task["seed"]
    with Timer() as t:
        res = ALGORITHMS[task["algorithm"]].run(inst, make_rng(seed), **task["params"])
    info = {k: v for k, v in task.items() if k != "out"}
    save_json({**info, "time_s": t.seconds, "n_evals": res.n_evals,
               "F": res.F.tolist(), "X": res.X.astype(int).tolist()}, Path(task["out"]))
    return f'{task["instance"]} {task["config"]} {task["algorithm"]} run {task["run"]}: ' \
           f'{t.seconds:.1f}s, {len(res.F)} solutions'


def build_tasks(args) -> list[dict]:
    exp = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    instances = args.instances or yaml.safe_load((ROOT / "configs/instances.yaml").read_text())["analysis"]
    configs = args.configs or list(exp["configurations"])
    algorithms = args.algorithms or exp["algorithms"]
    n_runs = args.runs or exp["n_runs"]
    tasks = []
    for inst in instances:
        for cfg in configs:
            params = exp["configurations"][cfg]
            for run in range(n_runs):
                # same seed for both algorithms in the same run
                seed = run_seed(exp["base_seed"], run)
                for alg in algorithms:
                    out = args.out / f"{inst}_{cfg}_{alg}_run{run:02d}.json"
                    if not out.exists():
                        tasks.append(dict(instance=inst, config=cfg, algorithm=alg, run=run,
                                          seed=seed, params=params, out=str(out)))
    return tasks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instances", nargs="+")
    p.add_argument("--configs", nargs="+")
    p.add_argument("--algorithms", nargs="+", choices=list(ALGORITHMS))
    p.add_argument("--runs", type=int, help="number of runs (default: n_runs in experiments.yaml)")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes")
    p.add_argument("--out", type=Path, default=ROOT / "results/raw")
    args = p.parse_args()

    tasks = build_tasks(args)
    print(f"{len(tasks)} runs to do")
    with ProcessPoolExecutor(args.jobs) as pool:
        for msg in pool.map(run_one, tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
