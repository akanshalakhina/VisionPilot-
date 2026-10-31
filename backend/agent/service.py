"""Run lifecycle: propose -> owner approves or rejects -> execute in the sandbox -> learn.

The agent can only create a proposal. Execution is a separate step that needs the owner's
approval, re-checks the guardrails, and then feeds the results back into the agent's memory.
"""
from __future__ import annotations

from dataclasses import asdict

from backend.adapters.base import CampaignPlan
from backend.adapters.sandbox import SandboxAdapter
from backend.agent.guardrails import ApprovalRequired, GuardrailViolation, check_budget_value, check_plan
from backend.agent.llm import LLMClient
from backend.agent.memory import BanditMemory, MemoryStore
from backend.agent.planner import AgentRunner
from backend.agent.targeting import UpliftTargeter
from backend.agent.tools import AgentServices, estimate_variants, variant_names
from backend.config import Settings
from backend.db import create_run, get_run, list_runs, log_event, update_run


class RunNotFound(LookupError):
    """No agent run has this id."""


class InvalidState(ValueError):
    """The run is not in a state that allows this step."""


class AgentService:
    def __init__(
        self,
        settings: Settings,
        sandbox: SandboxAdapter,
        llm: LLMClient | None = None,
        targeter: UpliftTargeter | None = None,
    ) -> None:
        self.settings = settings
        self.sandbox = sandbox
        self.llm = llm
        self.names = variant_names(sandbox.env.n_arms)
        self.bandit = BanditMemory(MemoryStore(settings.db_path), sandbox.env.n_arms)
        self.services = AgentServices(
            bandit=self.bandit,
            targeter=targeter or UpliftTargeter(settings.uplift_customers, settings.sim_seed),
            variant_names=self.names,
            max_budget=settings.max_total_budget,
            exploration_floor=settings.exploration_floor,
        )

    # ---- propose ---------------------------------------------------------

    def plan(self, goal: str, budget: float) -> dict:
        db = self.settings.db_path
        try:
            check_budget_value(budget, self.settings.max_total_budget)
        except GuardrailViolation as exc:
            log_event(db, "agent", "agent_plan_rejected", {"goal": goal, "budget": budget, "reason": str(exc)})
            raise

        outcome = AgentRunner(self.services, self.llm, self.settings.agent_max_steps).run(goal, budget)
        if outcome.plan is None:
            log_event(db, "agent", "agent_plan_rejected", {"goal": goal, "budget": budget, "reason": outcome.explanation})
            raise GuardrailViolation(outcome.explanation)

        grounding = {**outcome.grounding, **outcome.extras}
        run_id = create_run(db, goal, budget, outcome.planner, outcome.explanation, outcome.plan, grounding, outcome.trace)
        log_event(db, "agent", "agent_plan_proposed",
                  {"run_id": run_id, "planner": outcome.planner, "budget": budget, "grounding_ok": grounding["ok"]})
        return self.get(run_id)

    # ---- read ------------------------------------------------------------

    def state(self) -> dict:
        """What the dashboard shows beside the runs: settings and what the agent has learned so far."""
        estimates = estimate_variants(self.services)
        return {
            "planner": "llm" if self.llm is not None else "rule_based",
            "simulated": True,
            "max_total_budget": self.settings.max_total_budget,
            "exploration_floor": self.settings.exploration_floor,
            "credible_interval_percent": estimates["credible_interval_percent"],
            "variants": estimates["variants"],
            "history": self.bandit.history(),
        }

    def get(self, run_id: int) -> dict:
        run = get_run(self.settings.db_path, run_id)
        if run is None:
            raise RunNotFound(f"no agent run with id {run_id}")
        return run

    def list(self, limit: int = 20, include_trace: bool = True) -> list[dict]:
        runs = list_runs(self.settings.db_path, limit)
        if not include_trace:  # the trace is large, and lists do not need it
            runs = [{k: v for k, v in run.items() if k != "trace"} for run in runs]
        return runs

    # ---- approve / execute -----------------------------------------------

    def approve(self, run_id: int, approved: bool) -> dict:
        run = self.get(run_id)
        if run["status"] != "proposed":
            raise InvalidState(f"this plan is already {run['status']}")
        update_run(self.settings.db_path, run_id, "approved" if approved else "rejected")
        log_event(self.settings.db_path, "owner", "agent_plan_approved" if approved else "agent_plan_rejected_by_owner",
                  {"run_id": run_id})
        return self.get(run_id)

    def _campaign_plan(self, plan: dict) -> CampaignPlan:
        index = {name: i for i, name in enumerate(self.names)}
        total = sum(plan["allocation"].values())
        allocation = {index[name]: share / total for name, share in plan["allocation"].items()}
        return CampaignPlan(name=plan["name"], budget=plan["budget"], allocation=allocation)

    def execute(self, run_id: int) -> dict:
        db = self.settings.db_path
        run = self.get(run_id)
        if run["status"] == "executed":
            raise InvalidState("this plan was already executed")
        if run["status"] != "approved":
            raise ApprovalRequired(
                "the owner rejected this plan" if run["status"] == "rejected" else "the owner has not approved this plan"
            )

        plan = self._campaign_plan(run["plan"])
        try:
            plan.validate()
            check_plan(plan, self.settings.max_total_budget)
            result = self.sandbox.execute(plan)
        except ValueError as exc:
            log_event(db, "agent", "agent_plan_execution_blocked", {"run_id": run_id, "reason": str(exc)})
            raise

        self.bandit.update(result.clicks, result.impressions)
        shown = {i: result.impressions[i] for i in result.impressions if result.impressions[i] > 0}
        best = max(shown, key=lambda i: result.clicks[i] / shown[i]) if shown else None
        self.bandit.record_run({
            "run_id": run_id,
            "spend": result.spend,
            "total_clicks": sum(result.clicks.values()),
            "total_impressions": sum(result.impressions.values()),
            "best_variant": self.names[best] if best is not None else "none",
        })
        update_run(db, run_id, "executed", asdict(result))
        log_event(db, "agent", "agent_plan_executed", {"run_id": run_id, "result": asdict(result)})
        return self.get(run_id)
