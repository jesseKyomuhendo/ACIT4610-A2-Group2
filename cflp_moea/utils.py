"""Seeding, timing and saving results."""
from __future__ import annotations

import time

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

# TODO (T9): saving/loading results
