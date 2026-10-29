from __future__ import annotations

import math

from backend.adapters.base import CampaignPlan


class GuardrailViolation(ValueError):
    """A plan broke a safety rule."""


class ApprovalRequired(GuardrailViolation):
    """The owner has not approved the plan."""


def check_budget_value(budget: float, max_total_budget: float) -> float:
    """A budget must be a real positive number within the cap. NaN and infinity are refused:
    comparisons with NaN are always false, so a NaN would otherwise slip past every limit."""
    if not math.isfinite(budget) or budget <= 0:
        raise GuardrailViolation("the budget must be a positive number")
    if budget > max_total_budget:
        raise GuardrailViolation(f"budget {budget:g} exceeds the cap of {max_total_budget:g}")
    return float(budget)


def check_plan(plan: CampaignPlan, max_total_budget: float) -> None:
    check_budget_value(plan.budget, max_total_budget)


def require_approval(approved: bool) -> None:
    if not approved:
        raise ApprovalRequired("the owner has not approved this plan")
