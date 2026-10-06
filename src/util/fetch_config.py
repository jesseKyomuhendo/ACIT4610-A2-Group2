"""The only module that reads config.yaml.

Other modules import the constants below, for example:
    from src.util.fetch_config import N_RUNS, CONFIGURATIONS

Run `python -m src.util.fetch_config` from the project root to check that
config.yaml loads and to print every setting.
"""

from pathlib import Path

import yaml

# fetch_config.py is in <root>/src/util/, so the project root is two folders up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as file:
    _config = yaml.safe_load(file)

# Paths (converted to absolute paths so they work from any folder)
DATA_DIR = PROJECT_ROOT / _config["paths"]["data_dir"]
CHECKSUM_FILE = _config["paths"]["checksum_file"]
RESULTS_DIR = PROJECT_ROOT / _config["paths"]["results_dir"]
RAW_DIR = PROJECT_ROOT / _config["paths"]["raw_dir"]
TABLES_DIR = PROJECT_ROOT / _config["paths"]["tables_dir"]
FIGURES_DIR = PROJECT_ROOT / _config["paths"]["figures_dir"]

# Instances
INSTANCES = _config["instances"]                 # {"small": "cap41", ...}
OVERSIZED_DEMAND = _config["oversized_demand"]   # "report_infeasible" or "split"

# Experiment
N_RUNS = _config["experiment"]["n_runs"]
BASE_SEED = _config["experiment"]["base_seed"]

# Parameter configurations: {"C1": {"pop_size": 100, ...}, ...}
CONFIGURATIONS = _config["configurations"]

# Algorithm details
TOURNAMENT_SIZE = _config["algorithms"]["tournament_size"]
SPEA2_K = _config["algorithms"]["spea2_k"]       # None = sqrt(pop_size + archive_size)

# Hypervolume
HV_REFERENCE_POINT = _config["hypervolume"]["reference_point"]

# Statistics
STAT_TEST = _config["statistics"]["test"]
ALPHA = _config["statistics"]["alpha"]

# Plotting
PLOT_DPI = _config["plotting"]["dpi"]
FRONT_TO_PLOT = _config["plotting"]["front_to_plot"]
CONFIG_TO_PLOT = _config["plotting"]["config_to_plot"]


if __name__ == "__main__":
    print(f"Loaded {CONFIG_PATH}\n")
    for name, value in list(globals().items()):
        if name.isupper():
            print(f"{name} = {value}")