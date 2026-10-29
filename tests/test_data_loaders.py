import numpy as np
import pandas as pd
import pytest

from fakes import write_fake_avazu, write_fake_criteo, write_fake_open_bandit
from ml.bandit.ope import estimate_policy_value
from ml.bandit.policies import UniformRandom
from ml.data.avazu import HASHED_COLUMNS, REQUIRED_COLUMNS, load_avazu, time_split
from ml.data.criteo_uplift import FEATURES, load_criteo_uplift, split_uplift, summary
from ml.data.open_bandit import load_open_bandit, to_logged_data


# ---- Avazu ---------------------------------------------------------------

def test_avazu_loader_hashes_text_and_drops_id(tmp_path):
    path = write_fake_avazu(tmp_path / "train.csv")
    raw = pd.read_csv(path, dtype=str)
    df = load_avazu(path)

    assert "id" not in df.columns
    assert len(df) == len(raw)
    for col in HASHED_COLUMNS:
        assert df[col].dtype == "int64"
    # ids such as '1e000005' must not be merged by being read as numbers
    assert df["site_id"].nunique() == raw["site_id"].nunique() == 6
    assert {"day_key", "hour_of_day", "dow"} <= set(df.columns)
    assert df["hour_of_day"].between(0, 23).all()


def test_avazu_sampling_is_uniform_and_repeatable(tmp_path):
    path = write_fake_avazu(tmp_path / "train.csv", n=8000)
    a = load_avazu(path, sample_frac=0.25, seed=3)
    b = load_avazu(path, sample_frac=0.25, seed=3)
    assert 1500 < len(a) < 2500
    assert a.equals(b)
    assert a["day_key"].nunique() == 6  # every day stays represented


def test_avazu_missing_column_fails_clearly(tmp_path):
    path = write_fake_avazu(tmp_path / "train.csv")
    pd.read_csv(path).drop(columns=["banner_pos"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="banner_pos"):
        load_avazu(path)


def test_time_split_is_chronological_and_disjoint(tmp_path):
    df = load_avazu(write_fake_avazu(tmp_path / "train.csv"))
    train, valid, test = time_split(df, valid_days=1, test_days=1)
    assert train["day_key"].max() < valid["day_key"].min() <= valid["day_key"].max() < test["day_key"].min()
    assert len(train) + len(valid) + len(test) == len(df)


def test_time_split_needs_enough_days(tmp_path):
    df = load_avazu(write_fake_avazu(tmp_path / "train.csv", days=2))
    with pytest.raises(ValueError):
        time_split(df, valid_days=1, test_days=1)


def test_required_columns_are_documented():
    assert "click" in REQUIRED_COLUMNS and "hour" in REQUIRED_COLUMNS


# ---- Criteo uplift -------------------------------------------------------

def test_criteo_loader_reads_gzip_with_compact_dtypes(tmp_path):
    df = load_criteo_uplift(write_fake_criteo(tmp_path / "u.csv.gz"))
    assert len(df) == 4000
    assert all(df[f].dtype == "float32" for f in FEATURES)
    assert df["treatment"].dtype == "int8"
    s = summary(df)
    assert 0.8 < s["treatment_ratio"] < 0.9


def test_criteo_loader_handles_a_file_without_header(tmp_path):
    df = load_criteo_uplift(write_fake_criteo(tmp_path / "u.csv.gz", header=False))
    assert list(df.columns[:3]) == ["f0", "f1", "f2"]
    assert len(df) == 4000


def test_criteo_split_keeps_treatment_mix(tmp_path):
    df = load_criteo_uplift(write_fake_criteo(tmp_path / "u.csv.gz"))
    train, test = split_uplift(df, test_size=0.25, seed=1)
    assert len(train) + len(test) == len(df)
    assert abs(train["treatment"].mean() - test["treatment"].mean()) < 0.02


# ---- Open Bandit ---------------------------------------------------------

def test_open_bandit_loads_and_feeds_off_policy_evaluation(tmp_path):
    root = write_fake_open_bandit(tmp_path)
    df = load_open_bandit(root, "random", "all")
    assert df["timestamp"].is_monotonic_increasing

    log = to_logged_data(df)
    assert log.position == 1  # smallest position present
    assert log.n_arms == 8
    assert len(log.logged) == (df["position"] == 1).sum()
    assert np.allclose(log.logged.propensities, 1 / 8)

    # A uniform target on a uniform log has weight 1, so IPW must equal the plain click rate.
    est = estimate_policy_value(log.logged, UniformRandom(log.n_arms))
    assert est.ipw == pytest.approx(log.logged.rewards.mean())


def test_open_bandit_missing_file_and_columns_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_open_bandit(tmp_path)
    root = write_fake_open_bandit(tmp_path)
    path = root / "random" / "all" / "all.csv"
    pd.read_csv(path, index_col=0).drop(columns=["click"]).to_csv(path)
    with pytest.raises(ValueError, match="click"):
        load_open_bandit(root)
