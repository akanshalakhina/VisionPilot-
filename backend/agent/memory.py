"""What the agent remembers between runs: its belief about each ad, and a short run history."""
from __future__ import annotations

import numpy as np

from backend.db import memory_get, memory_set

HISTORY_LIMIT = 20


class MemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def get(self, key: str, default=None):
        return memory_get(self.db_path, key, default)

    def set(self, key: str, value) -> None:
        memory_set(self.db_path, key, value)


class BanditMemory:
    """A Beta(alpha, beta) belief about each ad's click rate, updated from executed runs.

    alpha - 1 is the clicks seen and beta - 1 the non-clicks seen, so a fresh
    memory (all ones) means nothing has been observed yet.
    """

    def __init__(self, store: MemoryStore, n_arms: int, business: str = "default") -> None:
        self.store = store
        self.n_arms = n_arms
        self._belief_key = f"bandit:{business}"
        self._history_key = f"history:{business}"

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        saved = self.store.get(self._belief_key)
        if not saved:
            return np.ones(self.n_arms), np.ones(self.n_arms)
        return np.array(saved["alpha"], dtype=float), np.array(saved["beta"], dtype=float)

    def update(self, clicks: dict[int, int], impressions: dict[int, int]) -> None:
        alpha, beta = self.state()
        for arm, shown in impressions.items():
            got = clicks.get(arm, 0)
            alpha[arm] += got
            beta[arm] += shown - got
        self.store.set(self._belief_key, {"alpha": alpha.tolist(), "beta": beta.tolist()})

    def thompson_shares(self, floor: float = 0.01, mc_samples: int = 4000, seed: int = 0) -> np.ndarray:
        """Budget shares: the chance each ad is the best one, with a minimum share so none is dropped.

        The seed is fixed, so the same belief always gives the same split.
        """
        if self.n_arms * floor > 1:
            raise ValueError("the exploration floor is too large for this many ads")
        alpha, beta = self.state()
        draws = np.random.default_rng(seed).beta(alpha, beta, size=(mc_samples, self.n_arms))
        freq = np.bincount(draws.argmax(axis=1), minlength=self.n_arms) / mc_samples
        return floor + (1.0 - self.n_arms * floor) * freq

    def history(self) -> list[dict]:
        return self.store.get(self._history_key, [])

    def record_run(self, summary: dict) -> None:
        self.store.set(self._history_key, [*self.history(), summary][-HISTORY_LIMIT:])
