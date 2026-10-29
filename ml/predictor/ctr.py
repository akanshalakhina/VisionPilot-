"""CTR models and evaluation metrics.

Baseline: logistic regression on one-hot features (the classic CTR baseline).
Main model: LightGBM gradient boosting with native categorical features.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.predictor.features import FeatureBuilder


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Average gap between predicted and observed click rate, over equal-sized groups.

    Equal-count groups are used because CTR predictions are small and bunch up,
    which makes equal-width bins nearly empty.
    """
    order = np.argsort(p)
    gaps = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            gaps += abs(float(p[idx].mean()) - float(y_true[idx].mean())) * len(idx)
    return gaps / len(p)


def evaluate(y_true: np.ndarray, p: np.ndarray, base_rate: float) -> dict:
    """AUC, log loss, normalised entropy (log loss relative to always predicting the base rate)
    and calibration error. Normalised entropy below 1 means the model beats the base rate."""
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true)
    base = min(max(base_rate, 1e-6), 1 - 1e-6)
    entropy = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    ll = float(log_loss(y, p, labels=[0, 1]))
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": ll,
        "normalized_entropy": ll / float(entropy),
        "ece": float(expected_calibration_error(y, p)),
        "mean_prediction": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def build_logreg(builder: FeatureBuilder, c: float = 1.0, seed: int = 0) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), builder.categorical_columns),
            ("num", StandardScaler(), builder.numeric_columns),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=c, max_iter=300, random_state=seed))])


def build_lightgbm(seed: int = 0):
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError('LightGBM is not installed. Run: pip install -e ".[ml]"') from exc
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
