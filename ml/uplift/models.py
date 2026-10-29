"""Uplift models: estimate how much a treatment (showing the ad) changes each person's outcome.

  ResponseScorer  baseline. Predicts who will convert anyway. It ignores the treatment,
                  so it cannot tell persuadable customers from sure things.
  SLearner        one model with the treatment as an extra feature.
  TLearner        one outcome model for treated and one for control; uplift is the gap.
  XLearner        T-learner plus a second stage that models imputed individual effects
                  (Kuenzel et al., 2019), blended with the treatment propensity.

All models expect a randomised experiment. The X-learner blend uses the treated
fraction as a constant propensity, which is right for a randomised trial only.
Outcomes must be binary (0/1) and treatment must be 0/1.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def default_classifier(seed: int = 0, max_iter: int = 100) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=100, random_state=seed
    )


def default_regressor(seed: int = 0, max_iter: int = 100) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=100, random_state=seed
    )


def _validate(X, y, t):
    X = np.asarray(X, dtype="float32")
    y = np.asarray(y).astype(int)
    t = np.asarray(t).astype(int)
    if not (len(X) == len(y) == len(t)):
        raise ValueError("X, y and treatment must have the same length")
    if not set(np.unique(t)) <= {0, 1}:
        raise ValueError("treatment must contain only 0 and 1")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("outcome must contain only 0 and 1")
    for group in (0, 1):
        mask = t == group
        if mask.sum() == 0:
            raise ValueError("need both treated and control rows")
        if len(np.unique(y[mask])) < 2:
            raise ValueError("the outcome needs both 0 and 1 within the treated group and within the control group")
    return X, y, t


def _p1(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


class ResponseScorer:
    """Ranks customers by P(outcome | features). NOT an uplift estimate, only a baseline score."""

    def __init__(self, classifier=None, seed: int = 0) -> None:
        self.classifier = classifier if classifier is not None else default_classifier(seed)

    def fit(self, X, y, t) -> "ResponseScorer":
        X, y, t = _validate(X, y, t)
        self.model_ = clone(self.classifier).fit(X, y)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return _p1(self.model_, np.asarray(X, dtype="float32"))


class SLearner:
    def __init__(self, classifier=None, seed: int = 0) -> None:
        self.classifier = classifier if classifier is not None else default_classifier(seed)

    def fit(self, X, y, t) -> "SLearner":
        X, y, t = _validate(X, y, t)
        self.model_ = clone(self.classifier).fit(np.column_stack([X, t]), y)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = np.asarray(X, dtype="float32")
        ones = np.ones((len(X), 1), dtype="float32")
        return _p1(self.model_, np.hstack([X, ones])) - _p1(self.model_, np.hstack([X, 0 * ones]))


class TLearner:
    def __init__(self, classifier=None, seed: int = 0) -> None:
        self.classifier = classifier if classifier is not None else default_classifier(seed)

    def fit(self, X, y, t) -> "TLearner":
        X, y, t = _validate(X, y, t)
        self.treated_ = clone(self.classifier).fit(X[t == 1], y[t == 1])
        self.control_ = clone(self.classifier).fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = np.asarray(X, dtype="float32")
        return _p1(self.treated_, X) - _p1(self.control_, X)


class XLearner:
    def __init__(self, classifier=None, regressor=None, seed: int = 0) -> None:
        self.classifier = classifier if classifier is not None else default_classifier(seed)
        self.regressor = regressor if regressor is not None else default_regressor(seed)

    def fit(self, X, y, t) -> "XLearner":
        X, y, t = _validate(X, y, t)
        x1, y1, x0, y0 = X[t == 1], y[t == 1], X[t == 0], y[t == 0]
        mu1 = clone(self.classifier).fit(x1, y1)
        mu0 = clone(self.classifier).fit(x0, y0)
        # Imputed individual effects: what the treated customer gained versus the control
        # model's guess, and what the control customer would have gained versus the treated model's guess.
        d1 = y1 - _p1(mu0, x1)
        d0 = _p1(mu1, x0) - y0
        self.tau1_ = clone(self.regressor).fit(x1, d1)
        self.tau0_ = clone(self.regressor).fit(x0, d0)
        self.propensity_ = float(t.mean())
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = np.asarray(X, dtype="float32")
        g = self.propensity_
        return g * self.tau0_.predict(X) + (1.0 - g) * self.tau1_.predict(X)
