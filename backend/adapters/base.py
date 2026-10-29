from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class AdapterDisabledError(RuntimeError):
    """Raised when an execution adapter is switched off."""


@dataclass
class CampaignPlan:
    name: str
    budget: float                 # total money to spend
    allocation: dict[int, float]  # ad index -> share of the budget (shares sum to 1)

    def validate(self) -> None:
        if not math.isfinite(self.budget) or self.budget <= 0:
            raise ValueError("budget must be a positive number")
        if not self.allocation:
            raise ValueError("allocation must not be empty")
        if any(not math.isfinite(share) or share < 0 for share in self.allocation.values()):
            raise ValueError("allocation shares must be finite and not negative")
        if not math.isclose(sum(self.allocation.values()), 1.0, abs_tol=1e-6):
            raise ValueError("allocation shares must sum to 1")


@dataclass
class ExecutionResult:
    adapter: str
    simulated: bool
    spend: float
    impressions: dict[int, int] = field(default_factory=dict)
    clicks: dict[int, int] = field(default_factory=dict)


class ExecutionAdapter(ABC):
    """Where an approved plan is carried out. The agent never talks to an ad platform directly."""

    name: str

    @abstractmethod
    def execute(self, plan: CampaignPlan) -> ExecutionResult: ...
