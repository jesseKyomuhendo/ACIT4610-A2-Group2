"""Seeding, timing and saving results."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def run_seed(base_seed: int, run: int) -> int:
    """Same seed for both MOEAs in the same run."""
    return base_seed + run


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class Timer:
    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._start


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def load_results(folder: Path) -> list[dict]:
    """All raw run results in a folder."""
    return [json.loads(p.read_text()) for p in sorted(Path(folder).glob("*.json"))]
