from __future__ import annotations

import numpy as np

from backend.adapters.base import CampaignPlan, ExecutionAdapter, ExecutionResult
from ml.simulator.bandit_env import AdEnvironment


class SandboxAdapter(ExecutionAdapter):
    """Runs a plan against the SYNTHETIC simulator. No real money, no real customers."""

    name = "sandbox"

    def __init__(self, env: AdEnvironment, cost_per_impression: float = 0.05) -> None:
        self.env = env
        self.cost_per_impression = cost_per_impression

    def execute(self, plan: CampaignPlan) -> ExecutionResult:
        plan.validate()
        impressions: dict[int, int] = {}
        clicks: dict[int, int] = {}
        for arm, share in plan.allocation.items():
            if not 0 <= arm < self.env.n_arms:
                raise ValueError(f"unknown ad index {arm}")
            n = int(plan.budget * share / self.cost_per_impression)
            impressions[arm] = n
            if n == 0:
                clicks[arm] = 0
                continue
            contexts = self.env.sample_contexts(n)
            outcomes = self.env.step(contexts, np.full(n, arm))
            clicks[arm] = int(outcomes.sum())
        spend = round(sum(impressions.values()) * self.cost_per_impression, 2)
        return ExecutionResult(
            adapter=self.name, simulated=True, spend=spend, impressions=impressions, clicks=clicks
        )
