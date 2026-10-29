"""Compare uplift targeting with plain response targeting on a randomised experiment.

Usage:
  python -m ml.uplift.experiment --synthetic --out reports/uplift_synthetic_demo.json
  python -m ml.uplift.experiment --data criteo-research-uplift-v2.1.csv.gz --sample-frac 0.1 \
      --outcome visit --out reports/uplift_criteo.json

Models: random order, response model (ignores treatment), S-, T- and X-learner.
Metrics are computed on a held-out test split. The bootstrap resamples the TEST rows
only, so the intervals show test-set noise but not training noise. Repeat with
several seeds before quoting a result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.criteo_uplift import load_criteo_uplift, split_uplift, summary
from ml.simulator.uplift_env import make_uplift_data
from ml.uplift.metrics import evaluate_scores, qini_coefficient
from ml.uplift.models import ResponseScorer, SLearner, TLearner, XLearner, default_classifier, default_regressor

MODEL_NAMES = ("random", "response", "s_learner", "t_learner", "x_learner")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c[0] == "f" and c[1:].isdigit()]


def run_experiment(
    df: pd.DataFrame,
    outcome: str = "visit",
    test_size: float = 0.25,
    seed: int = 0,
    n_boot: int = 100,
    max_iter: int = 100,
) -> dict:
    features = _feature_columns(df)
    train, test = split_uplift(df, test_size=test_size, seed=seed, outcome=outcome)
    x_tr, y_tr, t_tr = train[features].to_numpy(), train[outcome].to_numpy(), train["treatment"].to_numpy()
    x_te, y_te, t_te = test[features].to_numpy(), test[outcome].to_numpy(), test["treatment"].to_numpy()

    clf = default_classifier(seed, max_iter)
    reg = default_regressor(seed, max_iter)
    models = {
        "response": ResponseScorer(clf),
        "s_learner": SLearner(clf),
        "t_learner": TLearner(clf),
        "x_learner": XLearner(clf, reg),
    }
    scores = {"random": np.random.default_rng(seed).random(len(test))}
    for name, model in models.items():
        scores[name] = model.fit(x_tr, y_tr, t_tr).predict_uplift(x_te)

    result: dict = {
        "config": {"seed": seed, "outcome": outcome, "test_size": test_size, "n_boot": n_boot, "max_iter": max_iter},
        "data": {"train_rows": len(train), "test_rows": len(test), "features": features},
        "models": {name: evaluate_scores(y_te, t_te, s, seed) for name, s in scores.items()},
    }

    if "true_tau" in test.columns:  # only the synthetic generator knows the truth
        true_tau = test["true_tau"].reset_index(drop=True)
        for name in ("s_learner", "t_learner", "x_learner"):
            result["models"][name]["rank_corr_with_true_tau"] = float(
                pd.Series(scores[name]).corr(true_tau, method="spearman")
            )

    rng = np.random.default_rng(seed)
    boots = {name: [] for name in scores}
    for _ in range(n_boot):
        idx = rng.integers(0, len(test), len(test))
        for name, s in scores.items():
            boots[name].append(qini_coefficient(y_te[idx], t_te[idx], s[idx], seed))
    for name in scores:
        draws = np.array(boots[name])
        result["models"][name]["qini_ci95"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        if name != "response":
            diff = draws - np.array(boots["response"])
            result["models"][name]["qini_minus_response"] = {
                "mean": float(diff.mean()),
                "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                "share_of_resamples_better": float((diff > 0).mean()),
            }
    return result


def print_report(res: dict, synthetic: bool) -> None:
    if synthetic:
        print("SYNTHETIC DATA - results are simulated, not real campaign data\n")
    d, c = res["data"], res["config"]
    print(f"train rows {d['train_rows']:,}  test rows {d['test_rows']:,}  outcome '{c['outcome']}'  seed {c['seed']}\n")
    print(f"{'model':<10} {'Qini coef':>11} {'95% interval':>26} {'uplift@10%':>11} {'uplift@30%':>11}")
    for name in MODEL_NAMES:
        m = res["models"][name]
        lo, hi = m["qini_ci95"]
        print(
            f"{name:<10} {m['qini_coefficient']:>11.5f} {f'[{lo:.5f}, {hi:.5f}]':>26} "
            f"{m['uplift_at_10pct']:>11.4f} {m['uplift_at_30pct']:>11.4f}"
        )
    if "rank_corr_with_true_tau" in res["models"]["x_learner"]:
        print("\nRank correlation with the true effect (synthetic only):")
        for name in ("s_learner", "t_learner", "x_learner"):
            print(f"  {name:<10} {res['models'][name]['rank_corr_with_true_tau']:>7.3f}")
    print("\nDifference in Qini coefficient versus the response model (bootstrap over test rows)")
    for name in ("s_learner", "t_learner", "x_learner"):
        diff = res["models"][name]["qini_minus_response"]
        lo, hi = diff["ci95"]
        print(f"  {name:<10} mean {diff['mean']:>9.5f}  95% [{lo:.5f}, {hi:.5f}]  better in {diff['share_of_resamples_better']:.0%} of resamples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="Criteo uplift csv or csv.gz")
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic generator instead of --data")
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/uplift_demo.json")
    args = parser.parse_args()

    if args.synthetic:
        df = make_uplift_data(seed=args.seed)
        outcome = "visit"
    elif args.data:
        df = load_criteo_uplift(args.data, sample_frac=args.sample_frac, seed=args.seed)
        outcome = args.outcome
        print("Data summary:", json.dumps(summary(df), indent=2))
    else:
        parser.error("give --data PATH or --synthetic")

    res = run_experiment(df, outcome=outcome, seed=args.seed, n_boot=args.n_boot, max_iter=args.max_iter)
    res["synthetic"] = bool(args.synthetic)
    print_report(res, synthetic=args.synthetic)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
