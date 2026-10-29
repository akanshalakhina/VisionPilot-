"""Per-ad click model used by the direct-method and doubly-robust estimators."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class RewardModel:
    """One logistic regression per ad, fitted on the rows where that ad was shown."""

    def __init__(self, n_arms: int, c: float = 1.0) -> None:
        self.n_arms = n_arms
        self.c = c
        self._models: list = []

    def fit(self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> "RewardModel":
        global_mean = float(rewards.mean()) if len(rewards) else 0.0
        self._models = []
        for arm in range(self.n_arms):
            mask = actions == arm
            y = rewards[mask]
            if mask.sum() == 0:
                self._models.append(global_mean)
            elif len(np.unique(y)) < 2:
                self._models.append(float(y.mean()))
            else:
                model = LogisticRegression(C=self.c, max_iter=500)
                self._models.append(model.fit(contexts[mask], y))
        return self

    def predict(self, contexts: np.ndarray) -> np.ndarray:
        """Predicted click probability for every ad; shape (n, n_arms)."""
        out = np.zeros((len(contexts), self.n_arms))
        for arm, model in enumerate(self._models):
            if isinstance(model, float):
                out[:, arm] = model
            else:
                out[:, arm] = model.predict_proba(contexts)[:, 1]
        return out


def cross_fitted_predictions(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_arms: int,
    n_folds: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Predictions for each row from a model that never saw that row.

    Fitting and predicting on the same rows makes the direct-method estimate
    look better than it is. Cross-fitting avoids that.
    """
    n = len(actions)
    order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order, n_folds)
    q_hat = np.zeros((n, n_arms))
    for i, held_out in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        model = RewardModel(n_arms).fit(contexts[train], actions[train], rewards[train])
        q_hat[held_out] = model.predict(contexts[held_out])
    return q_hat
