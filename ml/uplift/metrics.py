"""Uplift evaluation from a randomised experiment (treatment assigned at random).

You never see both outcomes for one person, so uplift cannot be scored per row.
Instead rank customers by predicted uplift, then compare treated and control
people within the top slice of that ranking.

Definitions used here (other libraries normalise differently, so compare numbers
only within this codebase, on the same split):
  Qini curve  Q(k) = Y_T(k) - Y_C(k) * N_T(k) / N_C(k) for the top-k customers, where
              Y and N are outcome and head counts among treated (T) and control (C).
              It is divided by the population size so it reads as incremental
              outcomes per customer.
  AUUC        area under that curve over the targeting fraction 0..1.
  Qini coef.  area between the curve and the straight line you get from targeting
              people in random order. Positive means the ranking finds the people
              the treatment actually moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _validate(y, t, score):
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t).astype(int)
    score = np.asarray(score, dtype="float64")
    if not (len(y) == len(t) == len(score)):
        raise ValueError("y, treatment and score must have the same length")
    if t.sum() == 0 or t.sum() == len(t):
        raise ValueError("need both treated and control rows")
    return y, t, score


def _ranked(y, t, score, seed):
    # Random tie-break, so tied scores (for example a constant score) behave like random targeting.
    order = np.lexsort((np.random.default_rng(seed).random(len(score)), -score))
    return y[order], t[order]


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def qini_curve(y, t, score, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (fraction targeted, incremental outcomes per customer), both starting at 0."""
    y, t, score = _validate(y, t, score)
    y_s, t_s = _ranked(y, t, score, seed)
    n = len(y_s)
    n_t = np.cumsum(t_s)
    n_c = np.cumsum(1 - t_s)
    y_t = np.cumsum(y_s * t_s)
    y_c = np.cumsum(y_s * (1 - t_s))
    # Until the first control customer appears there is nothing to compare with, so count 0.
    ratio = np.divide(n_t, n_c, out=np.zeros(n), where=n_c > 0)
    gain = (y_t - y_c * ratio) / n
    frac = np.arange(1, n + 1) / n
    return np.concatenate([[0.0], frac]), np.concatenate([[0.0], gain])


def auuc(y, t, score, seed: int = 0) -> float:
    x, q = qini_curve(y, t, score, seed)
    return _trapezoid(q, x)


def qini_coefficient(y, t, score, seed: int = 0) -> float:
    x, q = qini_curve(y, t, score, seed)
    random_line = x * q[-1]
    return _trapezoid(q - random_line, x)


def uplift_at(y, t, score, fraction: float, seed: int = 0) -> float:
    """Observed treated-minus-control outcome rate among the top `fraction` of customers."""
    y, t, score = _validate(y, t, score)
    y_s, t_s = _ranked(y, t, score, seed)
    k = max(1, int(round(fraction * len(y_s))))
    top_y, top_t = y_s[:k], t_s[:k]
    if top_t.sum() == 0 or top_t.sum() == k:
        return float("nan")
    return float(top_y[top_t == 1].mean() - top_y[top_t == 0].mean())


def uplift_by_decile(y, t, score, seed: int = 0) -> pd.DataFrame:
    """Observed uplift in each tenth of the ranking, best first. A good model slopes downward."""
    y, t, score = _validate(y, t, score)
    y_s, t_s = _ranked(y, t, score, seed)
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(len(y_s)), 10), start=1):
        yy, tt = y_s[idx], t_s[idx]
        rate_t = float(yy[tt == 1].mean()) if (tt == 1).any() else float("nan")
        rate_c = float(yy[tt == 0].mean()) if (tt == 0).any() else float("nan")
        rows.append(
            {"decile": i, "n": int(len(idx)), "treated": int(tt.sum()),
             "rate_treated": rate_t, "rate_control": rate_c, "uplift": rate_t - rate_c}
        )
    return pd.DataFrame(rows)


def evaluate_scores(y, t, score, seed: int = 0) -> dict:
    deciles = uplift_by_decile(y, t, score, seed)
    return {
        "qini_coefficient": qini_coefficient(y, t, score, seed),
        "auuc": auuc(y, t, score, seed),
        "uplift_at_10pct": uplift_at(y, t, score, 0.10, seed),
        "uplift_at_30pct": uplift_at(y, t, score, 0.30, seed),
        "deciles": deciles.to_dict(orient="records"),
    }
