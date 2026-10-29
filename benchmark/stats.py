"""Small statistics helpers. Benchmarks with few tasks or seeds need honest intervals."""
from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Unlike the plain normal interval it stays
    inside 0..1 and behaves sensibly when every task passes."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lower, upper) of the mean, by resampling the values. Use it over seeds, not rows."""
    arr = np.asarray(values, dtype="float64")
    if len(arr) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(arr.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))
