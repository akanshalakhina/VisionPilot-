"""Loader for the Criteo Uplift Prediction Dataset (v2.1).

Source: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
License: CC BY-NC-SA 4.0 (Criteo Data Terms of Use). Non-commercial use with
attribution is fine for coursework. Cite: Diemert, Betlei, Renaudin, Amini,
"A Large Scale Benchmark for Uplift Modeling", AdKDD and TargetAd Workshop, KDD 2018.

Size: about 14 million rows, 297 MB compressed but about 3.2 GB uncompressed.
This loader reads the .gz file directly. Do NOT unzip it on a small disk. Prefer
running full experiments on Kaggle, and use `sample_frac` locally.

Columns: f0..f11 (anonymised floats), treatment (1 = eligible for targeting),
conversion, visit, exposure (1 = the ad was actually shown). Note that
treatment = 1 does not guarantee the person saw an ad.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [f"f{i}" for i in range(12)]
TREATMENT = "treatment"
OUTCOMES = ("visit", "conversion")
EXPOSURE = "exposure"
COLUMNS = [*FEATURES, TREATMENT, "conversion", "visit", EXPOSURE]


def load_criteo_uplift(
    path: str | Path,
    sample_frac: float = 1.0,
    nrows: int | None = None,
    seed: int = 0,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Read the uplift file (csv or csv.gz) as compact float32 / int8 columns."""
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1]")

    first = pd.read_csv(path, header=None, nrows=1)
    has_header = str(first.iloc[0, 0]).strip() == "f0"
    # If the file has no header row, the column order is assumed to be COLUMNS.
    kwargs = {"header": 0} if has_header else {"header": None, "names": COLUMNS}
    dtype = {c: "float32" for c in FEATURES}
    dtype.update({c: "int8" for c in (TREATMENT, "conversion", "visit", EXPOSURE)})

    rng = np.random.default_rng(seed)
    kept: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(path, dtype=dtype, chunksize=chunksize, **kwargs):
        if sample_frac < 1:
            chunk = chunk[rng.random(len(chunk)) < sample_frac]
        kept.append(chunk)
        total += len(chunk)
        if nrows is not None and total >= nrows:
            break

    df = pd.concat(kept, ignore_index=True)
    if nrows is not None:
        df = df.iloc[:nrows].reset_index(drop=True)

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected Criteo columns: {missing}")
    if not set(df[TREATMENT].unique()) <= {0, 1}:
        raise ValueError("treatment column must contain only 0 and 1")
    return df


def summary(df: pd.DataFrame) -> dict:
    """Headline numbers worth printing in a report: sizes, rates and the raw lift."""
    treated = df[df[TREATMENT] == 1]
    control = df[df[TREATMENT] == 0]
    out = {"rows": int(len(df)), "treatment_ratio": float(df[TREATMENT].mean())}
    for outcome in OUTCOMES:
        out[f"{outcome}_rate"] = float(df[outcome].mean())
        out[f"{outcome}_rate_treated"] = float(treated[outcome].mean())
        out[f"{outcome}_rate_control"] = float(control[outcome].mean())
        out[f"{outcome}_raw_lift"] = out[f"{outcome}_rate_treated"] - out[f"{outcome}_rate_control"]
    return out


def split_uplift(
    df: pd.DataFrame, test_size: float = 0.25, seed: int = 0, outcome: str = "visit"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split that keeps the treatment/outcome mix equal in both halves."""
    strata = df[TREATMENT].astype(int) * 2 + df[outcome].astype(int)
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)
    return train.reset_index(drop=True), test.reset_index(drop=True)
