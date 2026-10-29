"""Run a policy online against the simulator and track how much it loses to the best ad."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.bandit.policies import Policy
from ml.simulator.bandit_env import AdEnvironment


@dataclass
class OnlineResult:
    rewards: np.ndarray            # (n_rounds,) observed clicks
    cumulative_regret: np.ndarray  # (n_rounds,) expected clicks lost vs. always showing the best ad


def run_online(env: AdEnvironment, policy: Policy, n_rounds: int) -> OnlineResult:
    contexts = env.sample_contexts(n_rounds)
    true_probs = env.click_prob(contexts)
    best = true_probs.max(axis=1)
    rewards = np.zeros(n_rounds)
    regret = np.zeros(n_rounds)
    for t in range(n_rounds):
        action = policy.select(contexts[t])
        reward = float(env.step(contexts[t : t + 1], np.array([action]))[0])
        policy.update(contexts[t], action, reward)
        rewards[t] = reward
        regret[t] = best[t] - true_probs[t, action]
    return OnlineResult(rewards=rewards, cumulative_regret=np.cumsum(regret))
