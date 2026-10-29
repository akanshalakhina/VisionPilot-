"""Small FABRICATED files in the real dataset formats, used only to test the loaders.

Numbers measured on these files say nothing about real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def write_fake_avazu(path: Path, n: int = 8000, days: int = 6, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    day = rng.integers(0, days, n)
    hh = rng.integers(0, 24, n)
    banner = rng.choice([0, 1, 2, 3], n, p=[0.7, 0.25, 0.03, 0.02])
    # '1e000005' looks like a float to a careless CSV reader; it must stay text.
    site_ids = np.array(["1e000005", "00000123", "abcdef01", "deadbeef", "0badc0de", "5e7f0001"])
    site_cat = rng.choice(["28905ebd", "f028772b", "3e814130", "0569f928"], n)
    cat_effect = {"28905ebd": 0.0, "f028772b": 0.6, "3e814130": -0.5, "0569f928": 0.3}
    ips = np.array([f"{i:08x}" for i in range(300)])
    device_ip = ips[np.minimum(rng.zipf(1.6, n), 300) - 1]
    logit = (
        -1.8
        + 0.8 * (banner == 1)
        - 0.4 * (banner == 0)
        + np.array([cat_effect[c] for c in site_cat])
        + 0.4 * (hh >= 18)
    )
    df = pd.DataFrame(
        {
            "id": rng.integers(10**17, 10**19, n, dtype=np.uint64).astype(str),
            "click": (rng.random(n) < _sigmoid(logit)).astype(int),
            "hour": 14102100 + day * 100 + hh,
            "C1": rng.choice([1005, 1002, 1010], n),
            "banner_pos": banner,
            "site_id": rng.choice(site_ids, n),
            "site_domain": rng.choice(["d0001", "d0002", "d0003"], n),
            "site_category": site_cat,
            "app_id": rng.choice(["ecad2386", "92f5800b", "e2fcccd2"], n),
            "app_domain": rng.choice(["7801e8d9", "b9c8fd6b"], n),
            "app_category": rng.choice(["07d7df22", "0f2161f8"], n),
            "device_id": rng.choice(["a99f214a", "0f7c61dc", "c357dbff"], n),
            "device_ip": device_ip,
            "device_model": rng.choice(["8a4875bd", "1f0bc64f", "d787e91b"], n),
            "device_type": rng.choice([1, 0, 4], n),
            "device_conn_type": rng.choice([0, 2], n),
            "C14": rng.choice(range(15000, 15040), n),
            "C15": 320,
            "C16": rng.choice([50, 250], n),
            "C17": rng.choice(range(1700, 1720), n),
            "C18": rng.choice([0, 2, 3], n),
            "C19": rng.choice([35, 39, 167], n),
            "C20": rng.choice([-1, 100084, 100148], n),
            "C21": rng.choice([79, 23, 32], n),
        }
    )
    df.to_csv(path, index=False)
    return path


def write_fake_criteo(path: Path, n: int = 4000, seed: int = 0, header: bool = True) -> Path:
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, 12)).astype("float32")
    treatment = (rng.random(n) < 0.85).astype(int)
    exposure = treatment * (rng.random(n) < 0.03).astype(int)
    visit_p = 0.05 + 0.03 * treatment * (feats[:, 0] > 0)
    visit = (rng.random(n) < visit_p).astype(int)
    conversion = visit * (rng.random(n) < 0.06).astype(int)
    df = pd.DataFrame(feats, columns=[f"f{i}" for i in range(12)])
    df["treatment"], df["conversion"], df["visit"], df["exposure"] = treatment, conversion, visit, exposure
    df.to_csv(path, index=False, header=header, compression="gzip")
    return path


def write_fake_open_bandit(root: Path, n: int = 6000, n_arms: int = 8, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    folder = root / "random" / "all"
    folder.mkdir(parents=True, exist_ok=True)
    item = rng.integers(0, n_arms, n)
    user = {f"user_feature_{i}": rng.choice([f"u{i}{j:03x}" for j in range(5)], n) for i in range(4)}
    click_p = 0.003 + 0.01 * (item % 3)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2019-11-24", periods=n, freq="s").astype(str),
            "item_id": item,
            "position": rng.choice([1, 2, 3], n),
            "click": (rng.random(n) < click_p).astype(int),
            "propensity_score": 1.0 / n_arms,
            **user,
            "user-item_affinity_0": rng.random(n),
            "user-item_affinity_1": rng.random(n),
        }
    )
    df.to_csv(folder / "all.csv")  # keeps the unnamed leading index column
    pd.DataFrame({"item_feature_0": rng.random(n_arms), "item_feature_1": rng.integers(0, 3, n_arms)}).to_csv(
        folder / "item_context.csv"
    )
    return root
