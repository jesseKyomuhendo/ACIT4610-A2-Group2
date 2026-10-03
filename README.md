# ACIT4610-A2-Group2 — Multi-Objective Capacitated Facility Location (CFLP) with MOEAs

Two Multi-Objective Evolutionary Algorithms (planned: **NSGA-II** and **SPEA2**) minimise
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

## Run experiments
_To be added once the algorithms are implemented._

## Data source
J. E. Beasley, OR-Library — https://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html.
See `data/orlib/README.md` for the file format and field mapping.
