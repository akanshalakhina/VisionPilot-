"""Audience targeting with the uplift model.

The customer base here is SYNTHETIC: a randomised experiment from the simulator, with an
X-learner trained on the first part and the rest held out. Held-out customers are ranked,
and their observed outcomes give an honest check of the model's predictions.
"""
from __future__ import annotations

import numpy as np

from ml.simulator.uplift_env import make_uplift_data
from ml.uplift.metrics import uplift_at
from ml.uplift.models import XLearner, default_classifier, default_regressor

CUSTOMERS_BASIS = 1000  # results are quoted per this many customers


class UpliftTargeter:
    def __init__(self, n_customers: int = 20_000, seed: int = 0, max_iter: int = 60) -> None:
        self.n_customers = n_customers
        self.seed = seed
        self.max_iter = max_iter
        self._ready = False

    def _fit(self) -> None:
        df = make_uplift_data(n=self.n_customers, seed=self.seed)
        cut = int(0.7 * len(df))
        train, held = df.iloc[:cut], df.iloc[cut:]
        features = [c for c in df.columns if c[0] == "f" and c[1:].isdigit()]
        model = XLearner(
            default_classifier(self.seed, self.max_iter), default_regressor(self.seed, self.max_iter)
        ).fit(train[features].to_numpy(), train["visit"].to_numpy(), train["treatment"].to_numpy())
        self.tau_hat = model.predict_uplift(held[features].to_numpy())
        self.visit = held["visit"].to_numpy()
        self.treatment = held["treatment"].to_numpy()
        self._ready = True

    def summary(self, fraction: float) -> dict:
        """What targeting the top `fraction` of customers by predicted uplift is expected to give."""
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be above 0 and at most 1")
        if not self._ready:
            self._fit()
        n_total = len(self.tau_hat)
        n_target = max(1, int(round(fraction * n_total)))
        top = np.argsort(-self.tau_hat, kind="stable")[:n_target]
        observed = uplift_at(self.visit, self.treatment, self.tau_hat, fraction, seed=self.seed)
        return {
            "data_source": "synthetic customer base",
            "target_fraction": fraction,
            "customers_in_base": n_total,
            "customers_targeted": n_target,
            "customers_basis": CUSTOMERS_BASIS,
            "predicted_extra_visits_targeted": float(self.tau_hat[top].mean() * CUSTOMERS_BASIS),
            "predicted_extra_visits_everyone": float(self.tau_hat.mean() * CUSTOMERS_BASIS),
            "observed_extra_visits_targeted": None if np.isnan(observed) else float(observed * CUSTOMERS_BASIS),
            "note": (
                "Extra visits are per customers_basis customers. The observed figure comes from held-out "
                "customers in a randomised test and is noisy, so treat it as a rough check only."
            ),
        }
