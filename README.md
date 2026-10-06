# ACIT4610 Assignment 2 Multi-objective CFLP with NSGA-II and SPEA2

Group 2

This project solves the multi-objective Capacitated Facility Location Problem with two MOEAs written by hand, NSGA-II and SPEA2. Both algorithms minimise the facility-opening cost f1 and the customer-allocation cost f2 on the OR-Library benchmark instances, and they are compared with hypervolume, front size, runtime and a Wilcoxon signed-rank test.

## Requirements

- Python 3.10 or newer
- The packages in `requirements.txt`, which are numpy, matplotlib, pyyaml, ipykernel and notebook

## Installation

Open a terminal in the project root, the folder that contains `config.yaml`, and follow the steps below.

### *Step 1. Create the virtual environment*

Windows

```
python -m venv .venv
```

Mac and Linux

```
python3 -m venv .venv
```

### *Step 2. Activate the virtual environment*

Windows Command Prompt

```
.venv\Scripts\activate
```

Windows PowerShell

```
.venv\Scripts\Activate.ps1
```

Mac and Linux

```
source .venv/bin/activate
```

The terminal prompt now starts with `(.venv)`.

### *Step 3. Install the packages*

The same on every operating system.

```
pip install -r requirements.txt
```

## Project structure

```
ACIT4610-A2-Group2/
├── config.yaml                 every setting
├── training.ipynb              runs the whole pipeline
├── requirements.txt            Python packages
├── README.md
├── data/
│   └── orlib/
│       ├── cap41.txt ... cap122.txt    the six OR-Library instances
│       ├── capopt.txt                  kept for reference, not used
│       └── SHA256SUMS                  hashes that prove the data is unmodified
├── results/                    created by the pipeline
│   ├── raw/                    final front, runtime and seed of every run
│   ├── tables/                 results tables as CSV and PNG
│   └── figures/                Pareto-front plots
└── src/
    ├── util/
    │   └── fetch_config.py     the only module that reads config.yaml
    ├── data_loader.py          verifies, loads and prepares the instances
    ├── representation.py       chromosome, initialisation, crossover and mutation
    ├── repair.py               fixes capacity violations
    ├── objectives.py           f1, f2 and Pareto dominance
    ├── nsga2.py                NSGA-II
    ├── spea2.py                SPEA2
    ├── metrics.py              normalisation, hypervolume and front size
    ├── stats_tests.py          Wilcoxon signed-rank test
    ├── plotting.py             Pareto-front plots and results tables
    ├── run_experiments.py      runs all experiments and saves the raw results
    └── analyze_results.py      computes the metrics and saves the tables and plots
```

## Running the experiments

Start Jupyter from the project root with the virtual environment active.

```
jupyter notebook
```

Open `training.ipynb` in the browser and choose Run All. The notebook runs the whole pipeline in order.

1. Checks that the notebook uses the project's virtual environment
2. Loads `config.yaml` and prints every setting
3. Verifies that the data files are unmodified and checks every instance for feasibility
4. Runs NSGA-II and SPEA2 on every instance, configuration and run
5. Computes the metrics and statistical tests and saves the tables and plots
6. Shows the results tables and Pareto-front plots

Each step can also be run on its own from the project root.

```
python -m src.util.fetch_config
python -m src.data_loader
python -m src.run_experiments
python -m src.analyze_results
```

## Configuration

All settings are in `config.yaml`, including the instances, the number of runs, the random seeds, the three parameter configurations, the hypervolume reference point and the significance level. The code never needs to be changed to run a different setup.

Run k of both algorithms uses the seed `base_seed + k`, so every result can be reproduced.

## Data

The six instances cap41, cap42, cap101, cap102, cap121 and cap122 come unmodified from the OR-Library by J. E. Beasley. They are stored in `data/orlib`, and `SHA256SUMS` lets the code verify that they have not changed.

cap41 and cap42 have no feasible solution under the assignment's rules. The assignment requires every customer to be assigned to exactly one opened facility (`Σi xij = 1`) without exceeding that facility's capacity (`Σj dj xij ≤ Si yi`). Customers 11 and 34 have demands of 5,495 and 12,912, which exceed the capacity of 5,000 of every facility, so no single facility can serve them. The code detects this and reports both instances as infeasible instead of running the algorithms on them. Setting `oversized_demand` to `split` in `config.yaml` would split these customers into smaller parts served by several facilities, which would only be used if the assignment allowed it.

`capopt.txt` is kept for reference and is not used by the code.