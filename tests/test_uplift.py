import json

import numpy as np
import pytest

from fakes import write_fake_criteo
from ml.data.criteo_uplift import load_criteo_uplift
from ml.simulator.uplift_env import make_uplift_data
from ml.uplift.experiment import run_experiment
from ml.uplift.metrics import qini_coefficient, qini_curve, uplift_at, uplift_by_decile
from ml.uplift.models import ResponseScorer, SLearner, TLearner, XLearner

FEATURES = [f"f{i}" for i in range(6)]


@pytest.fixture(scope="module")
def data():
    df = make_uplift_data(n=30_000, seed=0)
    train, test = df.iloc[:20_000], df.iloc[20_000:]
    return train, test


def _xyt(df):
    return df[FEATURES].to_numpy(), df["visit"].to_numpy(), df["treatment"].to_numpy()


# ---- metrics -------------------------------------------------------------

def test_qini_curve_ends_at_the_treated_scaled_effect(data):
    _, test = data
    _, y, t = _xyt(test)
    x, q = qini_curve(y, t, np.random.default_rng(0).random(len(y)))
    assert x[0] == 0 and q[0] == 0 and x[-1] == 1
    expected = (t.sum() / len(t)) * (y[t == 1].mean() - y[t == 0].mean())
    assert q[-1] == pytest.approx(expected)


def test_uplift_at_full_population_is_the_overall_effect(data):
    _, test = data
    _, y, t = _xyt(test)
    score = np.random.default_rng(1).random(len(y))
    assert uplift_at(y, t, score, 1.0) == pytest.approx(y[t == 1].mean() - y[t == 0].mean())


def test_true_effect_ranking_beats_random_and_random_is_near_zero(data):
    _, test = data
    _, y, t = _xyt(test)
    best = qini_coefficient(y, t, test["true_tau"].to_numpy())
    random_scores = [qini_coefficient(y, t, np.random.default_rng(s).random(len(y)), seed=s) for s in range(5)]
    assert best > 0
    assert abs(np.mean(random_scores)) < 0.25 * best


def test_a_constant_score_behaves_like_random_targeting(data):
    _, test = data
    _, y, t = _xyt(test)
    best = qini_coefficient(y, t, test["true_tau"].to_numpy())
    assert abs(qini_coefficient(y, t, np.zeros(len(y)))) < 0.25 * best


def test_deciles_cover_everyone_and_slope_downward_for_a_good_ranking(data):
    _, test = data
    _, y, t = _xyt(test)
    table = uplift_by_decile(y, t, test["true_tau"].to_numpy())
    assert len(table) == 10 and table["n"].sum() == len(test)
    assert table["uplift"].iloc[:2].mean() > table["uplift"].iloc[-2:].mean()


def test_metrics_reject_one_sided_data():
    with pytest.raises(ValueError):
        qini_coefficient([0, 1, 1], [1, 1, 1], [0.1, 0.2, 0.3])


# ---- models --------------------------------------------------------------

def test_learners_recover_the_true_effect_and_beat_response_targeting(data):
    train, test = data
    x_tr, y_tr, t_tr = _xyt(train)
    x_te, y_te, t_te = _xyt(test)

    scores = {
        "response": ResponseScorer().fit(x_tr, y_tr, t_tr).predict_uplift(x_te),
        "t": TLearner().fit(x_tr, y_tr, t_tr).predict_uplift(x_te),
        "x": XLearner().fit(x_tr, y_tr, t_tr).predict_uplift(x_te),
    }
    q = {name: qini_coefficient(y_te, t_te, s) for name, s in scores.items()}

    # The synthetic effect depends on f2 only, while who visits anyway depends on f0 and f1.
    assert q["t"] > q["response"]
    assert q["x"] > q["response"]
    true_tau = test["true_tau"].to_numpy()
    assert np.corrcoef(scores["x"], true_tau)[0, 1] > 0.3
    assert np.corrcoef(scores["t"], true_tau)[0, 1] > 0.3


def test_s_learner_runs_and_returns_one_value_per_row(data):
    train, test = data
    x_tr, y_tr, t_tr = _xyt(train)
    out = SLearner().fit(x_tr, y_tr, t_tr).predict_uplift(test[FEATURES].to_numpy())
    assert out.shape == (len(test),) and np.isfinite(out).all()


def test_models_reject_bad_input(data):
    train, _ = data
    x, y, t = _xyt(train.iloc[:2000])
    with pytest.raises(ValueError, match="treatment"):
        TLearner().fit(x, y, np.full(len(y), 2))
    with pytest.raises(ValueError, match="treated and control"):
        TLearner().fit(x, y, np.ones(len(y), dtype=int))
    with pytest.raises(ValueError, match="outcome"):
        XLearner().fit(x, np.full(len(y), 3), t)


# ---- experiment runner ---------------------------------------------------

def test_experiment_runs_end_to_end_on_a_criteo_format_file(tmp_path):
    df = load_criteo_uplift(write_fake_criteo(tmp_path / "u.csv.gz", n=6000))
    res = run_experiment(df, outcome="visit", n_boot=5, max_iter=20)
    assert set(res["models"]) == {"random", "response", "s_learner", "t_learner", "x_learner"}
    for name, m in res["models"].items():
        assert len(m["deciles"]) == 10
        assert m["qini_ci95"][0] <= m["qini_ci95"][1]
    assert "qini_minus_response" in res["models"]["x_learner"]
    json.dumps(res)  # must be serialisable


def test_experiment_reports_the_true_effect_correlation_on_synthetic_data():
    res = run_experiment(make_uplift_data(n=12_000, seed=1), n_boot=5, max_iter=30)
    assert "rank_corr_with_true_tau" in res["models"]["x_learner"]
