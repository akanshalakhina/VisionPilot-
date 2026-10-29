"""Feature engineering for the CTR predictor.

Two ideas keep a model on 40-million-row ad data practical:
  * Rare category values are pooled into one 'unseen' code (0). Only values seen
    at least `min_count` times in the TRAINING data get their own code.
  * device_ip and device_id have millions of values, so the model uses how often
    each appears in the training data (log count) rather than the id itself.

Everything is fitted on the training split only, so nothing leaks from the
validation or test days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = [
    "C1", "banner_pos", "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category", "device_model", "device_type",
    "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
COUNT_COLUMNS = ["device_ip", "device_id"]
TIME_COLUMNS = ["hour_of_day", "dow"]


class FeatureBuilder:
    def __init__(self, min_count: int = 50, max_levels: int = 200) -> None:
        self.min_count = min_count
        self.max_levels = max_levels
        self.levels_: dict[str, pd.Index] = {}
        self.counts_: dict[str, pd.Series] = {}

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        for col in CATEGORICAL:
            counts = df[col].value_counts()
            counts = counts[counts >= self.min_count].nlargest(self.max_levels)
            self.levels_[col] = pd.Index(counts.index)
        self.counts_ = {col: df[col].value_counts() for col in COUNT_COLUMNS}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, np.ndarray] = {}
        for col in CATEGORICAL:
            # code 0 = rare or never seen in training, codes 1.. = common values
            out[col] = (self.levels_[col].get_indexer(df[col]) + 1).astype("int32")
        for col in TIME_COLUMNS:
            out[col] = df[col].to_numpy().astype("int32")
        for col in COUNT_COLUMNS:
            counts = df[col].map(self.counts_[col]).fillna(0).to_numpy()
            out[f"{col}_logcount"] = np.log1p(counts).astype("float32")
        return pd.DataFrame(out, index=df.index)

    @property
    def categorical_columns(self) -> list[str]:
        return [*CATEGORICAL, *TIME_COLUMNS]

    @property
    def numeric_columns(self) -> list[str]:
        return [f"{c}_logcount" for c in COUNT_COLUMNS]
