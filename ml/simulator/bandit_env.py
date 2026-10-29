"""Synthetic ad-selection environment with a known ground truth.

Each round a customer arrives with a feature vector (the context). The agent
picks one ad variant (an arm) and the customer clicks with probability
sigmoid(context @ theta[arm] + bias[arm]).

Because the true click probabilities are known, the exact value of any policy
can be computed. That lets us test off-policy estimators honestly: an estimate
made from logged data can be compared with the truth.

SYNTHETIC DATA: label every result from this module as simulated. It is a test
bench for the learning code, not evidence about real campaigns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ml.bandit.policies import Policy


@dataclass
class LoggedData:
    """Interaction log collected by a logging policy with known propensities."""

    contexts: np.ndarray      # (n, dim)
    actions: np.ndarray       # (n,) arm index that was shown
    rewards: np.ndarray       # (n,) 1 if clicked, else 0
    propensities: np.ndarray  # (n,) P(logged action | context) under the logging policy

    def __len__(self) -> int:
        return len(self.actions)


class AdEnvironment:
    def __init__(self, n_arms: int = 5, dim: int = 6, seed: int = 0,
                 bias_mean: float = -1.5, theta_scale: float = 1.0) -> None:
        """`bias_mean` sets the typical click rate (-1.5 gives roughly 25%, -4 gives roughly 2%) and
        `theta_scale` how strongly the customer context changes which ad works."""
        init_rng = np.random.default_rng(seed)
        self.n_arms = n_arms
        self.dim = dim
        self.theta = init_rng.normal(0.0, theta_scale, size=(n_arms, dim))
        self.bias = init_rng.normal(bias_mean, 0.5, size=n_arms)
        self._rng = np.random.default_rng(seed + 1)

    def sample_contexts(self, n: int) -> np.ndarray:
        return self._rng.normal(size=(n, self.dim))

    def click_prob(self, contexts: np.ndarray) -> np.ndarray:
        """True P(click | context, arm); shape (n, n_arms)."""
        logits = contexts @ self.theta.T + self.bias
        return 1.0 / (1.0 + np.exp(-logits))

    def step(self, contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Show `actions[i]` to customer `contexts[i]` and return click outcomes."""
        probs = self.click_prob(contexts)[np.arange(len(actions)), actions]
        return (self._rng.random(len(actions)) < probs).astype(float)

    def log(self, policy: "Policy", n: int) -> LoggedData:
        """Run `policy` for n customers and record what happened."""
        contexts = self.sample_contexts(n)
        probs = policy.action_probs(contexts)
        u = self._rng.random(n)[:, None]
        actions = np.minimum((probs.cumsum(axis=1) < u).sum(axis=1), self.n_arms - 1)
        propensities = probs[np.arange(n), actions]
        rewards = self.step(contexts, actions)
        return LoggedData(contexts, actions, rewards, propensities)

    def policy_value(self, policy: "Policy", n_eval: int = 50_000, eval_seed: int = 12345) -> float:
        """True expected click rate of `policy`, averaged over fresh customers."""
        contexts = np.random.default_rng(eval_seed).normal(size=(n_eval, self.dim))
        probs = policy.action_probs(contexts)
        return float((probs * self.click_prob(contexts)).sum(axis=1).mean())

    def oracle_value(self, n_eval: int = 50_000, eval_seed: int = 12345) -> float:
        """Click rate of an all-knowing policy that always shows the best ad."""
        contexts = np.random.default_rng(eval_seed).normal(size=(n_eval, self.dim))
        return float(self.click_prob(contexts).max(axis=1).mean())
