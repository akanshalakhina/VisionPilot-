"""Train and evaluate the CTR predictor on Avazu-format data.

Usage:
  python -m ml.predictor.train --data PATH/train.csv.gz --sample-frac 0.05 --out models/ctr

Protocol: split by day (train -> validation -> test), fit features on train only,
use the validation day for early stopping, and report final numbers on the test day.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from ml.data.avazu import load_avazu, time_split
from ml.predictor.ctr import build_lightgbm, build_logreg, evaluate
from ml.predictor.features import FeatureBuilder


def run_training(
    data_path: str | Path,
    models: tuple[str, ...] = ("logreg", "lightgbm"),
    sample_frac: float = 0.05,
    nrows: int | None = None,
    seed: int = 0,
    out_dir: str | Path = "models/ctr",
    valid_days: int = 1,
    test_days: int = 1,
    min_count: int = 50,
    max_levels: int = 200,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_avazu(data_path, sample_frac=sample_frac, nrows=nrows, seed=seed)
    train, valid, test = time_split(df, valid_days=valid_days, test_days=test_days)

    builder = FeatureBuilder(min_count=min_count, max_levels=max_levels).fit(train)
    x_train, x_valid, x_test = (builder.transform(d) for d in (train, valid, test))
    y_train, y_valid, y_test = (d["click"].to_numpy() for d in (train, valid, test))
    base_rate = float(y_train.mean())

    result: dict = {
        "data": {
            "source_file": Path(data_path).name,
            "sample_frac": sample_frac,
            "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
            "ctr": {"train": base_rate, "valid": float(y_valid.mean()), "test": float(y_test.mean())},
            "train_days": sorted(int(d) for d in train["day_key"].unique()),
            "valid_days": sorted(int(d) for d in valid["day_key"].unique()),
            "test_days": sorted(int(d) for d in test["day_key"].unique()),
        },
        "config": {"seed": seed, "min_count": min_count, "max_levels": max_levels},
        "models": {},
    }
    constant = np.full(len(y_test), base_rate)
    result["models"]["base_rate"] = {"test": evaluate(y_test, constant, base_rate)}

    joblib.dump(builder, out / "feature_builder.joblib")

    if "logreg" in models:
        t0 = time.time()
        model = build_logreg(builder, seed=seed).fit(x_train, y_train)
        result["models"]["logreg"] = {
            "fit_seconds": round(time.time() - t0, 1),
            "valid": evaluate(y_valid, model.predict_proba(x_valid)[:, 1], base_rate),
            "test": evaluate(y_test, model.predict_proba(x_test)[:, 1], base_rate),
        }
        joblib.dump(model, out / "logreg.joblib")

    if "lightgbm" in models:
        import lightgbm as lgb

        t0 = time.time()
        model = build_lightgbm(seed=seed)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="binary_logloss",
            categorical_feature=builder.categorical_columns,
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        result["models"]["lightgbm"] = {
            "fit_seconds": round(time.time() - t0, 1),
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            "valid": evaluate(y_valid, model.predict_proba(x_valid)[:, 1], base_rate),
            "test": evaluate(y_test, model.predict_proba(x_test)[:, 1], base_rate),
        }
        joblib.dump(model, out / "lightgbm.joblib")

    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def print_report(result: dict) -> None:
    d = result["data"]
    print(f"Rows  train={d['rows']['train']:,}  valid={d['rows']['valid']:,}  test={d['rows']['test']:,}")
    print(f"Train days {d['train_days']}  valid {d['valid_days']}  test {d['test_days']}")
    print(f"\n{'model':<10} {'AUC':>7} {'logloss':>9} {'NE':>7} {'ECE':>8}   (test split)")
    for name, row in result["models"].items():
        m = row["test"]
        print(f"{name:<10} {m['auc']:>7.4f} {m['logloss']:>9.4f} {m['normalized_entropy']:>7.4f} {m['ece']:>8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="Avazu train.csv or train.csv.gz")
    parser.add_argument("--models", nargs="+", default=["logreg", "lightgbm"], choices=["logreg", "lightgbm"])
    parser.add_argument("--sample-frac", type=float, default=0.05)
    parser.add_argument("--nrows", type=int, default=None, help="smoke tests only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/ctr")
    args = parser.parse_args()

    result = run_training(
        args.data, models=tuple(args.models), sample_frac=args.sample_frac,
        nrows=args.nrows, seed=args.seed, out_dir=args.out,
    )
    print_report(result)
    print(f"\nSaved {Path(args.out) / 'metrics.json'}")


if __name__ == "__main__":
    main()
