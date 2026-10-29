"""Loader for the Open Bandit Dataset (ZOZOTOWN fashion e-commerce).

Source: https://research.zozo.com/data.html  (paper: Saito, Aihara, Matsutani, Narita,
arXiv:2008.07146). The data page does not state a dataset license, so check the
terms yourself before relying on it, and cite the paper.

Expected layout (the same one the `obp` package reads):
    <data_dir>/<behavior_policy>/<campaign>/<campaign>.csv
with behavior_policy in {random, bts} and campaign in {all, men, women}.
The CSV has a leading index column. Columns used: timestamp, item_id, position,
click, propensity_score and every column containing 'user_feature'.

Verified against the real files? NOT YET. This loader follows the documentation
and the `obp` source, and is tested on a small fabricated file with that layout.
Run it on a real sample first and fix any difference in the header.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.simulator.bandit_env import LoggedData

REQUIRED_COLUMNS = ["timestamp", "item_id", "position", "click", "propensity_score"]


@dataclass
class OpenBanditLog:
    logged: LoggedData
    n_arms: int
    position: int  # the slot that was kept


def load_open_bandit(
    data_dir: str | Path, behavior_policy: str = "random", campaign: str = "all"
) -> pd.DataFrame:
    path = Path(data_dir) / behavior_policy / campaign / f"{campaign}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Expected <data_dir>/<policy>/<campaign>/<campaign>.csv")
    df = pd.read_csv(path, index_col=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    if not any("user_feature" in c for c in df.columns):
        raise ValueError(f"{path} has no 'user_feature' columns")
    return df.sort_values("timestamp").reset_index(drop=True)


def to_logged_data(df: pd.DataFrame, position: int | None = None, max_levels: int = 10) -> OpenBanditLog:
    """Turn Open Bandit rows into the LoggedData format used by our off-policy evaluation.

    Each recommendation slot has its own propensity, so we keep ONE slot
    (`position`, default the smallest value present) and treat each row as a
    single-ad decision. Customer features are one-hot encoded, keeping the
    `max_levels` most common values per feature and pooling the rest as 'other'.
    """
    if position is None:
        position = int(df["position"].min())
    rows = df[df["position"] == position]
    if rows.empty:
        raise ValueError(f"no rows with position == {position}")
    if (rows["propensity_score"] <= 0).any():
        raise ValueError("propensity_score must be positive for off-policy evaluation")

    user_cols = [c for c in rows.columns if "user_feature" in c]
    encoded = {}
    for col in user_cols:
        values = rows[col].astype(str)
        top = values.value_counts().nlargest(max_levels).index
        encoded[col] = values.where(values.isin(top), "other")
    contexts = pd.get_dummies(pd.DataFrame(encoded), drop_first=True, dtype=float).to_numpy()

    actions = rows["item_id"].to_numpy().astype(int)
    logged = LoggedData(
        contexts=contexts,
        actions=actions,
        rewards=rows["click"].to_numpy().astype(float),
        propensities=rows["propensity_score"].to_numpy().astype(float),
    )
    return OpenBanditLog(logged=logged, n_arms=int(actions.max()) + 1, position=position)
