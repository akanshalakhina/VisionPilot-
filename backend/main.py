from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.adapters.base import CampaignPlan
from backend.adapters.meta import MetaAdapter
from backend.adapters.sandbox import SandboxAdapter
from backend.agent.guardrails import ApprovalRequired, GuardrailViolation, check_plan, require_approval
from backend.agent.llm import LLMClient, build_llm
from backend.agent.service import AgentService, InvalidState, RunNotFound
from backend.agent.targeting import UpliftTargeter
from backend.config import Settings
from backend.db import init_db, list_events, log_event
from ml.simulator.bandit_env import AdEnvironment


class RunRequest(BaseModel):
    name: str
    budget: float = Field(allow_inf_nan=False)
    allocation: dict[int, float]
    approved: bool = False


class PlanRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=500)
    budget: float = Field(allow_inf_nan=False)


class ApprovalBody(BaseModel):
    approved: bool


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    targeter: UpliftTargeter | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    init_db(settings.db_path)

    sandbox = SandboxAdapter(AdEnvironment(seed=settings.sim_seed))
    meta = MetaAdapter(enabled=settings.meta_adapter_enabled)
    llm = llm if llm is not None else build_llm(settings)  # fails fast if the LLM settings are incomplete
    agent = AgentService(settings, sandbox, llm=llm, targeter=targeter)

    app = FastAPI(title="VisionPilot", version="0.2.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "adapter": sandbox.name,
            "meta_adapter_enabled": meta.enabled,
            "planner": "llm" if llm is not None else "rule_based",
        }

    # ---- direct sandbox run (no agent) ----------------------------------

    @app.post("/sandbox/run")
    def run_in_sandbox(req: RunRequest) -> dict:
        """Run an owner-approved plan in the simulator. Rejections are logged too."""
        plan = CampaignPlan(name=req.name, budget=req.budget, allocation=req.allocation)
        try:
            plan.validate()
            check_plan(plan, settings.max_total_budget)
            require_approval(req.approved)
            result = sandbox.execute(plan)
        except ApprovalRequired as exc:
            log_event(settings.db_path, "api", "plan_rejected", {"plan": asdict(plan), "reason": str(exc)})
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            log_event(settings.db_path, "api", "plan_rejected", {"plan": asdict(plan), "reason": str(exc)})
            raise HTTPException(status_code=422, detail=str(exc))
        log_event(settings.db_path, "api", "plan_executed", {"plan": asdict(plan), "result": asdict(result)})
        return asdict(result)

    # ---- the agent ------------------------------------------------------

    @app.post("/agent/plan")
    def agent_plan(req: PlanRequest) -> dict:
        """The agent proposes a plan. Nothing runs until the owner approves it."""
        try:
            return agent.plan(req.goal, req.budget)
        except GuardrailViolation as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/agent/runs")
    def agent_runs(limit: int = 20) -> list[dict]:
        return agent.list(limit)

    @app.get("/agent/runs/{run_id}")
    def agent_run(run_id: int) -> dict:
        try:
            return agent.get(run_id)
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/agent/runs/{run_id}/approve")
    def agent_approve(run_id: int, body: ApprovalBody) -> dict:
        try:
            return agent.approve(run_id, body.approved)
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except InvalidState as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/agent/runs/{run_id}/execute")
    def agent_execute(run_id: int) -> dict:
        try:
            return agent.execute(run_id)
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ApprovalRequired, InvalidState) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/audit")
    def audit(limit: int = 50) -> list[dict]:
        return list_events(settings.db_path, limit=limit)

    return app
