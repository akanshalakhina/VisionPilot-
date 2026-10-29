"""Off-policy evaluation: estimate how a new policy would perform using only logged data.

This is how the agent can judge a budget strategy without spending real money on it.

Estimators:
  IPW    reweight each logged click by pi(a|x) / logging_propensity. Unbiased, high variance.
  SNIPW  IPW with self-normalised weights. Slightly biased, much more stable.
  DM     direct method: trust a learned click model. Low variance, biased if the model is wrong.
  DR     doubly robust: DM plus an IPW correction on the model's residuals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ml.bandit.policies import Policy
from ml.bandit.reward_model import cross_fitted_predictions
from ml.simulator.bandit_env import LoggedData


@dataclass
class OpeEstimates:
    ipw: float
    snipw: float
    dm: float
    dr: float
    effective_sample_size: float  # how many logged rows effectively support the estimate

    def as_dict(self) -> dict:
        return asdict(self)


def estimate_policy_value(
    logged: LoggedData,
    target: Policy,
    n_folds: int = 2,
    seed: int = 0,
    clip: float | None = None,
) -> OpeEstimates:
    n = len(logged)
    idx = np.arange(n)
    pi = target.action_probs(logged.contexts)
    weights = pi[idx, logged.actions] / logged.propensities
    if clip is not None:
        weights = np.minimum(weights, clip)
    rewards = logged.rewards

    ipw = float(np.mean(weights * rewards))
    weight_sum = float(weights.sum())
    snipw = float((weights * rewards).sum() / weight_sum) if weight_sum > 0 else float("nan")

    q_hat = cross_fitted_predictions(
        logged.contexts, logged.actions, rewards, target.n_arms, n_folds=n_folds, seed=seed
    )
    model_value = (pi * q_hat).sum(axis=1)
    dm = float(model_value.mean())
    dr = float(np.mean(model_value + weights * (rewards - q_hat[idx, logged.actions])))

    sq = float((weights ** 2).sum())
    ess = weight_sum ** 2 / sq if sq > 0 else 0.0
    return OpeEstimates(ipw=ipw, snipw=snipw, dm=dm, dr=dr, effective_sample_size=ess)
