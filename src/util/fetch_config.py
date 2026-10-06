"""The only module that reads config.yaml and turns every setting into a constant.

Run it as a module from the project root to print all settings.
"""

from pathlib import Path

import yaml

# This file is in root/src/util, so the project root is two folders up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as file:
    _config = yaml.safe_load(file)

# Paths are made absolute so they work from any folder
DATA_DIR = PROJECT_ROOT / _config["paths"]["data_dir"]
CHECKSUM_FILE = _config["paths"]["checksum_file"]
RESULTS_DIR = PROJECT_ROOT / _config["paths"]["results_dir"]
RAW_DIR = PROJECT_ROOT / _config["paths"]["raw_dir"]
TABLES_DIR = PROJECT_ROOT / _config["paths"]["tables_dir"]
FIGURES_DIR = PROJECT_ROOT / _config["paths"]["figures_dir"]

# Instances
INSTANCES = _config["instances"]                 # size category mapped to instance name

# Experiment
N_RUNS = _config["experiment"]["n_runs"]
BASE_SEED = _config["experiment"]["base_seed"]

# Configuration name mapped to its parameters
CONFIGURATIONS = _config["configurations"]

# Algorithm details
TOURNAMENT_SIZE = _config["algorithms"]["tournament_size"]
SPEA2_K = _config["algorithms"]["spea2_k"]       # None means sqrt(pop_size + archive_size)

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