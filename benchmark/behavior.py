"""Behaviour benchmark: run every task through the real agent service and check the stored results.

Each task gets a fresh database and agent, so what the agent remembers never leaks between
tasks. `warmup_runs` executes approved campaigns first, on purpose, so the agent has data.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Callable

from backend.adapters.sandbox import SandboxAdapter
from backend.agent.guardrails import GuardrailViolation
from backend.agent.service import AgentService
from backend.agent.targeting import UpliftTargeter
from backend.config import Settings
from backend.db import init_db
from benchmark.checks import check_plan, check_rejection
from benchmark.stats import wilson_interval
from benchmark.tasks import CATEGORIES, Task
from ml.simulator.bandit_env import AdEnvironment

BENCH_CAP = 5000.0
SECRET = "SECRET-TEST-KEY-do-not-leak"  # set as the API key so any leak into stored data is detected
_ids = count(1)


@dataclass
class TaskResult:
    task_id: str
    category: str
    checks: dict[str, bool] = field(default_factory=dict)
    crashed: str | None = None
    llm_involved: bool = False
    raw_text_rejected: bool = False
    template_fallback: bool = False
    tool_errors: int = 0

    @property
    def passed(self) -> bool:
        return self.crashed is None and bool(self.checks) and all(self.checks.values())


def run_task(task: Task, targeter: UpliftTargeter, llm_factory: Callable | None, workdir: str | Path) -> TaskResult:
    db = str(Path(workdir) / f"task_{next(_ids)}.db")
    settings = Settings(db_path=db, max_total_budget=BENCH_CAP, uplift_customers=targeter.n_customers,
                        llm_api_key=SECRET)
    init_db(db)
    svc = AgentService(settings, SandboxAdapter(AdEnvironment(seed=0)), llm=None, targeter=targeter)
    result = TaskResult(task.id, task.category)
    try:
        for _ in range(task.warmup_runs):  # the offline planner does the warm-up, never the LLM under test
            warm = svc.plan("warm-up campaign", 500.0)
            svc.approve(warm["id"], True)
            svc.execute(warm["id"])

        svc.llm = llm_factory() if llm_factory else None
        result.llm_involved = svc.llm is not None
        run, rejected = None, False
        try:
            run = svc.plan(task.goal, task.budget)
        except GuardrailViolation:
            rejected = True

        n_variants = len(svc.names)
        if task.expect == "plan":
            result.checks = check_plan(task, run, db, SECRET, n_variants) if run else {"plan_created": False}
        else:
            result.checks = check_rejection(task, db, SECRET, rejected and run is None)
        if run:
            result.raw_text_rejected = "first_llm_text_rejected" in run["grounding"]
            result.template_fallback = result.llm_involved and run["grounding"]["explanation_source"] == "template"
            result.tool_errors = sum(not step["ok"] for step in run["trace"])
    except Exception as exc:  # a crash is a failed task, not a crashed benchmark
        result.crashed = f"{type(exc).__name__}: {exc}"
    return result


def run_suite(tasks: list[Task], targeter: UpliftTargeter, llm_factory: Callable | None,
              workdir: str | Path) -> list[TaskResult]:
    return [run_task(t, targeter, llm_factory, workdir) for t in tasks]


def summarise(results: list[TaskResult]) -> dict:
    def block(subset: list[TaskResult]) -> dict:
        passed = sum(r.passed for r in subset)
        lo, hi = wilson_interval(passed, len(subset))
        return {"n": len(subset), "passed": passed, "rate": passed / len(subset) if subset else float("nan"),
                "ci95": [lo, hi]}

    llm_runs = [r for r in results if r.llm_involved]
    return {
        "overall": block(results),
        "by_category": {c: block([r for r in results if r.category == c]) for c in CATEGORIES
                        if any(r.category == c for r in results)},
        "check_failures": dict(Counter(k for r in results for k, ok in r.checks.items() if not ok)),
        "failed_tasks": [
            {"task": r.task_id, "failed_checks": [k for k, ok in r.checks.items() if not ok], "crashed": r.crashed}
            for r in results if not r.passed
        ],
        "llm_runs": len(llm_runs),
        "raw_llm_text_rejected_rate": (sum(r.raw_text_rejected for r in llm_runs) / len(llm_runs)) if llm_runs else None,
        "template_fallback_rate": (sum(r.template_fallback for r in llm_runs) / len(llm_runs)) if llm_runs else None,
        "tool_errors": sum(r.tool_errors for r in results),
    }
