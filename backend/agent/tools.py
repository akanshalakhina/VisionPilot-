"""The tools the agent can call.

There is deliberately NO tool that spends money or runs anything. The agent can only
propose a plan, and the budget split inside that plan always comes from the bandit,
never from numbers the LLM supplies. Running a plan needs the owner's approval through
the API.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from scipy.stats import beta as beta_dist

from backend.agent.guardrails import GuardrailViolation, check_budget_value
from backend.agent.memory import BanditMemory
from backend.agent.targeting import UpliftTargeter

EXPLORATION_FLOOR = 0.01  # must match Settings.exploration_floor (a test checks this)
CREDIBLE_INTERVAL_PERCENT = 90
DEFAULT_TARGET_FRACTION = 0.3


def variant_names(n_arms: int) -> list[str]:
    if n_arms > 26:
        raise ValueError("at most 26 ad variants are supported")
    return [f"Ad {chr(ord('A') + i)}" for i in range(n_arms)]


@dataclass
class AgentServices:
    bandit: BanditMemory
    targeter: UpliftTargeter
    variant_names: list[str]
    max_budget: float
    exploration_floor: float = EXPLORATION_FLOOR


def estimate_variants(services: AgentServices) -> dict:
    """Click-rate estimate per ad from the agent's learned belief. Ads with no results have null estimates.
    Shared by the estimate tool and the dashboard's state endpoint, so both always agree."""
    alpha, beta = services.bandit.state()
    tail = (100 - CREDIBLE_INTERVAL_PERCENT) / 200
    rows = []
    for i, name in enumerate(services.variant_names):
        impressions = int(round(alpha[i] + beta[i] - 2))
        if impressions == 0:
            rows.append({"name": name, "impressions_seen": 0, "clicks_seen": 0,
                         "estimated_ctr": None, "ctr_low": None, "ctr_high": None})
            continue
        rows.append({
            "name": name,
            "impressions_seen": impressions,
            "clicks_seen": int(round(alpha[i] - 1)),
            "estimated_ctr": float(alpha[i] / (alpha[i] + beta[i])),
            "ctr_low": float(beta_dist.ppf(tail, alpha[i], beta[i])),
            "ctr_high": float(beta_dist.ppf(1 - tail, alpha[i], beta[i])),
        })
    return {"credible_interval_percent": CREDIBLE_INTERVAL_PERCENT, "variants": rows,
            "note": "Estimates come from simulated results only. A null estimate means no results yet."}


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


class ToolBox:
    """One per agent run. Every call is recorded so the run can be audited and checked."""

    def __init__(self, services: AgentServices, owner_budget: float) -> None:
        self.s = services
        self.owner_budget = float(owner_budget)
        self.trace: list[dict] = []
        self.plan: dict | None = None
        self._tools: dict[str, Callable[..., dict]] = {
            "list_ad_variants": self._list_ad_variants,
            "estimate_variant_ctr": self._estimate_variant_ctr,
            "recall_past_runs": self._recall_past_runs,
            "recommend_budget_split": self._recommend_budget_split,
            "rank_audience": self._rank_audience,
            "propose_plan": self._propose_plan,
        }

    # ---- plumbing --------------------------------------------------------

    def schemas(self) -> list[dict]:
        budget = {"type": "number", "description": "Total budget the owner gave"}
        fraction = {"type": "number", "description": "Share of customers to target, above 0 and at most 1"}
        return [
            _schema("list_ad_variants", "List the ad variants that can be tested.", {}, []),
            _schema(
                "estimate_variant_ctr",
                "Estimated click rate of each ad from past results, with a credible interval. "
                "Ads with no results yet have no estimate.",
                {}, [],
            ),
            _schema("recall_past_runs", "Summaries of the most recent executed plans.", {}, []),
            _schema(
                "recommend_budget_split",
                "Split a budget across the ads with Thompson sampling.",
                {"total_budget": budget}, ["total_budget"],
            ),
            _schema(
                "rank_audience",
                "What targeting the top share of customers by predicted uplift is expected to give.",
                {"target_fraction": fraction}, ["target_fraction"],
            ),
            _schema(
                "propose_plan",
                "Propose the final plan for the owner to approve. Call it exactly once. "
                "The budget split is computed here from the bandit and cannot be overridden.",
                {
                    "total_budget": budget,
                    "target_fraction": fraction,
                    "plan_name": {"type": "string", "description": "Short name for the plan"},
                },
                ["total_budget"],
            ),
        ]

    def call(self, name: str, args: dict) -> dict:
        step = f"T{len(self.trace) + 1}"
        function = self._tools.get(name)
        if function is None:
            result, ok = {"error": f"unknown tool '{name}'"}, False
        else:
            try:
                result, ok = function(**(args or {})), True
            except TypeError:
                result, ok = {"error": f"invalid arguments for {name}"}, False
            except ValueError as exc:  # includes GuardrailViolation
                result, ok = {"error": str(exc)}, False
        self.trace.append({"id": step, "tool": name, "args": args, "ok": ok, "result": result})
        return result

    def _check_budget(self, total_budget: float) -> float:
        return check_budget_value(float(total_budget), self.s.max_budget)

    # ---- tools -----------------------------------------------------------

    def _list_ad_variants(self) -> dict:
        return {"variants": list(self.s.variant_names), "variant_count": len(self.s.variant_names)}

    def _estimate_variant_ctr(self) -> dict:
        return estimate_variants(self.s)

    def _recall_past_runs(self) -> dict:
        runs = self.s.bandit.history()[-3:]
        if not runs:
            return {"runs": [], "note": "No plan has been executed yet."}
        return {"runs": runs}

    def _split(self, total_budget: float) -> list[dict]:
        shares = self.s.bandit.thompson_shares(floor=self.s.exploration_floor)
        return [
            {"name": name, "share": float(share), "amount": round(total_budget * float(share), 2)}
            for name, share in zip(self.s.variant_names, shares)
        ]

    def _recommend_budget_split(self, total_budget: float) -> dict:
        total_budget = self._check_budget(total_budget)
        return {"total_budget": total_budget, "method": "thompson_sampling",
                "exploration_floor": self.s.exploration_floor, "split": self._split(total_budget)}

    def _rank_audience(self, target_fraction: float) -> dict:
        return self.s.targeter.summary(float(target_fraction))

    def _propose_plan(self, total_budget: float, target_fraction: float = DEFAULT_TARGET_FRACTION,
                      plan_name: str = "Campaign plan") -> dict:
        total_budget = self._check_budget(total_budget)
        if abs(total_budget - self.owner_budget) > 1e-6:
            raise GuardrailViolation(f"the plan budget must equal the owner's budget of {self.owner_budget:g}")
        target_fraction = float(target_fraction)
        if not 0 < target_fraction <= 1:
            raise ValueError("target_fraction must be above 0 and at most 1")
        # Digits are removed so a name can never smuggle an unchecked figure into the explanation.
        name = " ".join(re.sub(r"\d+", " ", str(plan_name)).split())[:60] or "Campaign plan"
        split = self._split(total_budget)
        self.plan = {
            "name": name,
            "budget": total_budget,
            "allocation": {row["name"]: row["share"] for row in split},
            "amounts": {row["name"]: row["amount"] for row in split},
            "target_fraction": target_fraction,
            "method": "thompson_sampling",
            "simulated": True,
        }
        return {"plan": self.plan, "status": "proposed", "needs_owner_approval": True}
