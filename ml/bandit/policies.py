"""Ad-selection policies.

A policy maps a customer context to a choice of ad variant. Every policy exposes
`action_probs`, the probability of choosing each ad, because off-policy
evaluation needs it to reweight logged data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ml.bandit.reward_model import RewardModel

if TYPE_CHECKING:
    from ml.simulator.bandit_env import LoggedData


def one_hot(indices: np.ndarray, n_arms: int) -> np.ndarray:
    probs = np.zeros((len(indices), n_arms))
    probs[np.arange(len(indices)), indices] = 1.0
    return probs


class Policy(ABC):
    """A (possibly stochastic) rule that picks an ad for each customer."""

    def __init__(self, n_arms: int, seed: int = 0) -> None:
        self.n_arms = n_arms
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def action_probs(self, contexts: np.ndarray) -> np.ndarray:
        """P(ad | context) for each row; shape (n, n_arms)."""

    def select(self, context: np.ndarray) -> int:
        probs = self.action_probs(context[None, :])[0]
        return int(self._rng.choice(self.n_arms, p=probs))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Learn from one observed outcome. Static policies ignore this."""


class UniformRandom(Policy):
    """Shows every ad equally often. The baseline and the logging policy."""

    def action_probs(self, contexts: np.ndarray) -> np.ndarray:
        return np.full((len(contexts), self.n_arms), 1.0 / self.n_arms)


class BernoulliThompson(Policy):
    """Thompson sampling with a Beta(alpha, beta) belief about each ad's click rate.

    It ignores the customer context, so it learns which ad is best on average.
    """

    def __init__(self, n_arms: int, seed: int = 0, mc_samples: int = 2000) -> None:
        super().__init__(n_arms, seed)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.mc_samples = mc_samples

    def action_probs(self, contexts: np.ndarray) -> np.ndarray:
        # There is no closed form, so estimate each ad's selection probability by simulation.
        draws = self._rng.beta(self.alpha, self.beta, size=(self.mc_samples, self.n_arms))
        freq = np.bincount(draws.argmax(axis=1), minlength=self.n_arms) / self.mc_samples
        return np.tile(freq, (len(contexts), 1))

    def select(self, context: np.ndarray) -> int:
        return int(self._rng.beta(self.alpha, self.beta).argmax())

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self.alpha[action] += reward
        self.beta[action] += 1.0 - reward


class LinUCB(Policy):
    """Contextual bandit: a ridge regression per ad plus an optimism bonus (disjoint LinUCB)."""

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, ridge: float = 1.0, seed: int = 0) -> None:
        super().__init__(n_arms, seed)
        self.alpha = alpha
        d = dim + 1  # one extra feature so each ad can learn its own baseline click rate
        self.a_inv = np.stack([np.eye(d) / ridge for _ in range(n_arms)])
        self.b = np.zeros((n_arms, d))

    @staticmethod
    def _phi(contexts: np.ndarray) -> np.ndarray:
        return np.hstack([contexts, np.ones((len(contexts), 1))])

    def _scores(self, contexts: np.ndarray) -> np.ndarray:
        x = self._phi(contexts)
        theta = np.einsum("kde,ke->kd", self.a_inv, self.b)
        mean = x @ theta.T
        var = np.einsum("nd,kde,ne->nk", x, self.a_inv, x)
        return mean + self.alpha * np.sqrt(np.maximum(var, 0.0))

    def action_probs(self, contexts: np.ndarray) -> np.ndarray:
        return one_hot(self._scores(contexts).argmax(axis=1), self.n_arms)

    def select(self, context: np.ndarray) -> int:
        return int(self._scores(context[None, :])[0].argmax())

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = np.append(context, 1.0)
        a_inv = self.a_inv[action]
        ax = a_inv @ x
        # Sherman-Morrison update of the inverse, so no matrix inversion per step
        self.a_inv[action] = a_inv - np.outer(ax, ax) / (1.0 + x @ ax)
        self.b[action] += reward * x


class LogisticGreedy(Policy):
    """Offline policy: fit one click model per ad on logged data, then show the best predicted ad."""

    def __init__(self, n_arms: int, seed: int = 0) -> None:
        super().__init__(n_arms, seed)
        self.model = RewardModel(n_arms)

    def fit(self, logged: "LoggedData") -> "LogisticGreedy":
        self.model.fit(logged.contexts, logged.actions, logged.rewards)
        return self

    def action_probs(self, contexts: np.ndarray) -> np.ndarray:
        return one_hot(self.model.predict(contexts).argmax(axis=1), self.n_arms)
