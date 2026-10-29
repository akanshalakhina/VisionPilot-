"""Synthetic randomised experiment with a KNOWN individual treatment effect.

The ad changes a customer's chance of visiting by tau(x), which can be positive
(persuadable) or negative (the ad backfires). Customers with a high baseline
chance of visiting anyway are driven by different features than the ones that
decide who is persuadable. A model that only predicts who will visit therefore
targets the wrong people. Real data may behave differently, and the Criteo
experiment is what tells you that.

SYNTHETIC DATA: label every result from this module as simulated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_uplift_data(
    n: int = 30_000, dim: int = 6, treat_prob: float = 0.5, effect: float = 0.12, seed: int = 0
) -> pd.DataFrame:
    """Columns f0..f{dim-1}, treatment, visit (the outcome), plus true_tau for diagnostics only."""
    if dim < 3:
        raise ValueError("dim must be at least 3")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    baseline = 1.0 / (1.0 + np.exp(-(-1.5 + 0.9 * x[:, 0] + 0.5 * x[:, 1])))
    tau = effect * np.tanh(2.0 * x[:, 2])
    p_treated = np.clip(baseline + tau, 0.0, 1.0)
    treatment = (rng.random(n) < treat_prob).astype("int8")
    p = np.where(treatment == 1, p_treated, baseline)
    visit = (rng.random(n) < p).astype("int8")

    df = pd.DataFrame(x.astype("float32"), columns=[f"f{i}" for i in range(dim)])
    df["treatment"] = treatment
    df["visit"] = visit
    df["true_tau"] = (p_treated - baseline).astype("float32")
    return df
