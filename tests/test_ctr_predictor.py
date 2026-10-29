import json

import numpy as np
import pytest

from fakes import write_fake_avazu
from ml.data.avazu import load_avazu, time_split
from ml.predictor.ctr import evaluate, expected_calibration_error
from ml.predictor.features import FeatureBuilder
from ml.predictor.train import run_training


def test_metrics_reward_a_good_predictor():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.02, 0.6, 20_000)
    y = (rng.random(20_000) < true_p).astype(int)
    base = float(y.mean())

    good = evaluate(y, true_p, base)
    constant = evaluate(y, np.full(len(y), base), base)

    assert good["auc"] > 0.7
    assert constant["auc"] == pytest.approx(0.5)
    assert constant["normalized_entropy"] == pytest.approx(1.0, abs=0.01)
    assert good["normalized_entropy"] < 1.0
    assert good["ece"] < 0.02  # predictions equal the true probabilities, so they are calibrated


def test_calibration_error_detects_overconfidence():
    y = np.array([0, 1] * 500)
    assert expected_calibration_error(y, np.full(1000, 0.9)) == pytest.approx(0.4, abs=0.01)


def test_feature_builder_uses_training_data_only(tmp_path):
    df = load_avazu(write_fake_avazu(tmp_path / "train.csv"))
    train, _, test = time_split(df)
    builder = FeatureBuilder(min_count=5).fit(train)

    # Code 0 is reserved for values never seen (or too rare) in training.
    held_out = test.copy()
    held_out["site_domain"] = 123456789
    codes = builder.transform(held_out)["site_domain"]
    assert (codes == 0).all()

    seen = builder.transform(train)["site_domain"]
    assert seen.min() >= 1
    assert list(builder.transform(train).columns)[-2:] == ["device_ip_logcount", "device_id_logcount"]


def test_logreg_training_end_to_end_on_fabricated_data(tmp_path):
    data = write_fake_avazu(tmp_path / "train.csv", n=9000)
    out = tmp_path / "ctr"
    result = run_training(data, models=("logreg",), sample_frac=1.0, out_dir=out, min_count=5)

    assert (out / "metrics.json").exists() and (out / "logreg.joblib").exists()
    assert json.loads((out / "metrics.json").read_text())["models"]["logreg"]["test"]["auc"] > 0.6
    assert result["models"]["logreg"]["test"]["normalized_entropy"] < 1.0
    rows = result["data"]["rows"]
    assert rows["train"] > rows["valid"] and rows["train"] > rows["test"]


def test_lightgbm_training_end_to_end_on_fabricated_data(tmp_path):
    pytest.importorskip("lightgbm")
    data = write_fake_avazu(tmp_path / "train.csv", n=9000)
    result = run_training(data, models=("lightgbm",), sample_frac=1.0, out_dir=tmp_path / "ctr", min_count=5)
    row = result["models"]["lightgbm"]
    assert row["test"]["auc"] > 0.6
    assert row["best_iteration"] >= 1
