"""Loader for the Avazu click-through-rate dataset (Kaggle: avazu-ctr-prediction).

Columns follow the public competition description: id, click, hour (YYMMDDHH),
C1, banner_pos, site_*, app_*, device_*, C14-C21. Check them against the file you
download. The loader fails with a clear message if a required column is missing.

Terms: read the rules on the Kaggle competition page before using the data, and
cite the dataset in your report. Do not commit the data to git.

Memory: text columns are replaced by a 64-bit hash of their value. Category
identity is kept, and a few million rows fit comfortably in memory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HASHED_COLUMNS = [
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model",
]
INT_COLUMNS = [
    "C1", "banner_pos", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
REQUIRED_COLUMNS = ["click", "hour", *HASHED_COLUMNS, *INT_COLUMNS]


def _hash_strings(series: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(series, index=False).to_numpy().astype("int64")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_key (YYMMDD), hour_of_day and dow (0 = Monday) from the YYMMDDHH `hour` column."""
    hour = df["hour"].to_numpy().astype("int64")
    day_key = hour // 100
    dow_by_day = {
        int(d): pd.Timestamp(year=2000 + int(d) // 10_000, month=(int(d) // 100) % 100, day=int(d) % 100).day_of_week
        for d in np.unique(day_key)
    }
    df["day_key"] = day_key.astype("int32")
    df["hour_of_day"] = (hour % 100).astype("int8")
    df["dow"] = pd.Series(day_key).map(dow_by_day).to_numpy().astype("int8")
    return df


def load_avazu(
    path: str | Path,
    sample_frac: float = 1.0,
    nrows: int | None = None,
    seed: int = 0,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Read Avazu train data (csv or csv.gz), optionally keeping a random fraction of rows.

    `sample_frac` samples uniformly across the whole file, so every day stays represented.
    `nrows` keeps only the first N rows AFTER sampling. Use it for quick smoke tests only.
    """
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1]")

    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path} is missing expected Avazu columns: {missing}")

    dtype: dict = {c: "int32" for c in INT_COLUMNS}
    dtype.update({"click": "int8", "hour": "int64"})
    dtype.update({c: str for c in HASHED_COLUMNS})  # ids like '1e000005' must stay text
    usecols = [c for c in header if c != "id"]

    rng = np.random.default_rng(seed)
    kept: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunksize):
        if sample_frac < 1:
            chunk = chunk[rng.random(len(chunk)) < sample_frac]
        for col in HASHED_COLUMNS:
            chunk[col] = _hash_strings(chunk[col])
        kept.append(chunk)
        total += len(chunk)
        if nrows is not None and total >= nrows:
            break

    df = pd.concat(kept, ignore_index=True)
    if nrows is not None:
        df = df.iloc[:nrows].reset_index(drop=True)
    return add_time_features(df)


def time_split(
    df: pd.DataFrame, valid_days: int = 1, test_days: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by calendar day: earliest days train, then validation, then the latest days test.

    A random split would let the model see the future. Ad data drifts, so the
    honest question is how well a model trained on the past predicts later days.
    """
    days = np.sort(df["day_key"].unique())
    if len(days) < valid_days + test_days + 1:
        raise ValueError(f"need at least {valid_days + test_days + 1} distinct days, found {len(days)}")
    test_set = days[-test_days:]
    valid_set = days[-(test_days + valid_days):-test_days]
    train = df[~df["day_key"].isin(np.concatenate([valid_set, test_set]))]
    valid = df[df["day_key"].isin(valid_set)]
    test = df[df["day_key"].isin(test_set)]
    return train, valid, test
