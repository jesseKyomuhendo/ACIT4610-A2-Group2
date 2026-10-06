# ACIT4610-A2-Group2 — Multi-Objective Capacitated Facility Location (CFLP) with MOEAs

Two Multi-Objective Evolutionary Algorithms, **NSGA-II** and **SPEA2**, minimise
(1) facility-opening cost and (2) customer-allocation cost on OR-Library CFLP instances.

## Repository structure
```
data/orlib/            OR-Library instances (unmodified) + SHA256SUMS + capopt.txt
configs/               instances.yaml, experiments.yaml (3 parameter configurations)
cflp_moea/
  data_loader.py       parse cap*.txt files
  representation.py    chromosome encoding + feasibility repair/decoding (shared)
  objectives.py        f1, f2 evaluation + evaluation counter
  operators.py         initialisation, crossover, mutation (shared)
  algorithms/          common.py, nsga2.py, spea2.py
  metrics.py           normalisation, hypervolume, #non-dominated
  stats.py             statistical tests
  plotting.py          Pareto-front plots
  utils.py             seeding, timing, I/O
scripts/               verify_data.py, run_experiments.py, analyze_results.py, make_plots.py
tests/                 pytest unit tests
notebooks/             exploration notebooks
results/               raw/, tables/, figures/ (generated)
report/                report figures
```

## Installation
Requires Python 3.10+.
```bash
git clone https://github.com/jesseKyomuhendo/ACIT4610-A2-Group2.git
cd ACIT4610-A2-Group2
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Verify the data
```bash
python scripts/verify_data.py
pytest
```

## Run the experiments
All settings are in `configs/` (three configurations A, B, C; 10 runs; seeds 4610-4619).
Both MOEAs use the same configuration, seeds, representation, repair and operators.

```bash
# 1. all runs: 2 MOEAs x 3 configurations x 3 instances x 10 runs (about 40 min)
python scripts/run_experiments.py
#    faster on several cores, e.g. 8 processes (runtimes then include machine load)
python scripts/run_experiments.py --jobs 8
#    quick check: 2 runs of configuration A on cap101 only
python scripts/run_experiments.py --runs 2 --configs A --instances cap101 --out results/quick

# 2. tables: HV, number of non-dominated solutions, runtime, Mann-Whitney U tests
python scripts/analyze_results.py

# 3. figures: Pareto fronts and HV boxplots
python scripts/make_plots.py
```

The raw results of our full run are already in `results/raw/`, so steps 2 and 3 work
straight after cloning. Runs that already have a result file are skipped; delete
`results/raw/*.json` to run everything again. Fixed seeds give identical fronts on
every run; only runtimes vary between machines.

## Outputs
| Path | Content |
|------|---------|
| `results/raw/*.json` | one file per run: settings, seed, runtime, evaluations, final front |
| `results/tables/summary.md` | mean, std, best, worst HV; non-dominated solutions; mean runtime |
| `results/tables/tests.csv` | Mann-Whitney U p-value and A12 effect size per instance/configuration |
| `results/tables/normalisation.json` | objective bounds and HV reference point per instance |
| `results/figures/pareto_<instance>.png` | final fronts of both MOEAs on the same axes |
| `results/figures/hv_<instance>.png` | HV boxplots over the 10 runs |

## Method in short
- **Chromosome:** binary vector, `y[i] = 1` if facility i is open.
- **Repair/decoding** (`representation.py`, shared): open more facilities if capacity
  is too small, assign customers greedily (largest first, cheapest facility with room),
  improve by moving/swapping customers, close unused facilities.
- **cap41/cap42:** two customers are bigger than every facility (5000); only those two
  may be split over several facilities. See `data/orlib/README.md`.
- **Objectives:** f1 = sum of fixed costs of open facilities; f2 = sum of `C_ij` of the
  assignment (not multiplied by demand).
- **Hypervolume:** objectives scaled to [0, 1] with the min/max over all runs of both
  MOEAs per instance; reference point (1.1, 1.1).

## Data source
J. E. Beasley, OR-Library — https://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html.
See `data/orlib/README.md` for the file format and field mapping.
