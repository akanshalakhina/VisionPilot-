import json

import pytest
from fastapi.testclient import TestClient

from backend.adapters.sandbox import SandboxAdapter
from backend.agent.grounding import check_grounding
from backend.agent.guardrails import ApprovalRequired, GuardrailViolation
from backend.agent.llm import LLMError, LLMResponse, ToolCall
from backend.agent.service import AgentService, InvalidState
from backend.agent.targeting import UpliftTargeter
from backend.config import Settings
from backend.db import init_db, list_events
from backend.main import create_app
from ml.simulator.bandit_env import AdEnvironment

GOAL = "Get more customers for my festive sale"


@pytest.fixture(scope="module")
def targeter():
    return UpliftTargeter(n_customers=4000, seed=0, max_iter=20)


def make_service(tmp_path, targeter, llm=None, cap=5000.0):
    settings = Settings(db_path=str(tmp_path / "t.db"), max_total_budget=cap, uplift_customers=4000)
    init_db(settings.db_path)
    return AgentService(settings, SandboxAdapter(AdEnvironment(seed=0)), llm=llm, targeter=targeter), settings


class ScriptedLLM:
    """Plays back prepared replies. A step may be a function of the messages so far."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step


class BrokenLLM:
    def chat(self, messages, tools=None):
        raise LLMError("LLM request failed with HTTP 500")


TOOL_STEP = LLMResponse(tool_calls=[
    ToolCall("c1", "estimate_variant_ctr", {}),
    ToolCall("c2", "recommend_budget_split", {"total_budget": 1000}),
    ToolCall("c3", "rank_audience", {"target_fraction": 0.3}),
    ToolCall("c4", "propose_plan", {"total_budget": 1000, "target_fraction": 0.3, "plan_name": "Festive sale 2026"}),
])
BAD_TEXT = LLMResponse(content="Expect roughly a 41.37% click rate.")


def honest(messages):
    results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    row = next(r for r in results if "split" in r)["split"][0]
    return LLMResponse(content=f"I suggest giving {row['name']} {row['share'] * 100:.1f}% of the budget, about ₹{row['amount']:,.2f}.")


# ---- offline rule-based planner -------------------------------------------

def test_rule_based_plan_is_valid_and_fully_grounded(tmp_path, targeter):
    svc, _ = make_service(tmp_path, targeter)
    run = svc.plan(GOAL, 1000)

    assert run["status"] == "proposed" and run["planner"] == "rule_based"
    shares = list(run["plan"]["allocation"].values())
    assert sum(shares) == pytest.approx(1.0) and min(shares) >= 0.01 - 1e-9
    assert "minimum of 1%" in run["explanation"]
    assert run["plan"]["budget"] == 1000 and run["plan"]["simulated"] is True
    assert run["grounding"]["ok"] and run["grounding"]["explanation_source"] == "template"
    assert "Nothing runs until you approve" in run["explanation"]
    assert [s["id"] for s in run["trace"]][:2] == ["T1", "T2"]
    assert run["trace"][-1]["tool"] == "propose_plan"


def test_the_default_floor_is_one_percent_in_every_place_it_is_defined():
    from backend.agent.memory import BanditMemory
    from backend.agent.tools import EXPLORATION_FLOOR
    import inspect

    assert Settings().exploration_floor == EXPLORATION_FLOOR == 0.01
    assert inspect.signature(BanditMemory.thompson_shares).parameters["floor"].default == 0.01


def test_the_exploration_floor_is_configurable_and_reported(tmp_path, targeter):
    settings = Settings(db_path=str(tmp_path / "f.db"), max_total_budget=5000.0, uplift_customers=4000,
                        exploration_floor=0.1)
    init_db(settings.db_path)
    svc = AgentService(settings, SandboxAdapter(AdEnvironment(seed=0)), targeter=targeter)
    run = svc.plan(GOAL, 1000)
    assert min(run["plan"]["allocation"].values()) >= 0.1 - 1e-9
    assert "minimum of 10%" in run["explanation"] and run["grounding"]["ok"]


def test_a_budget_over_the_cap_is_rejected_and_logged(tmp_path, targeter):
    svc, settings = make_service(tmp_path, targeter, cap=500.0)
    with pytest.raises(GuardrailViolation):
        svc.plan(GOAL, 1000)
    assert list_events(settings.db_path)[0]["action"] == "agent_plan_rejected"
    assert svc.list() == []


# ---- approval and learning ------------------------------------------------

def test_a_plan_cannot_run_without_approval(tmp_path, targeter):
    svc, _ = make_service(tmp_path, targeter)
    run_id = svc.plan(GOAL, 1000)["id"]
    with pytest.raises(ApprovalRequired):
        svc.execute(run_id)

    assert svc.approve(run_id, False)["status"] == "rejected"
    with pytest.raises(ApprovalRequired, match="rejected"):
        svc.execute(run_id)
    with pytest.raises(InvalidState):
        svc.approve(run_id, True)


def test_the_agent_learns_from_an_executed_plan(tmp_path, targeter):
    svc, settings = make_service(tmp_path, targeter)
    first = svc.plan(GOAL, 1000)
    svc.approve(first["id"], True)
    done = svc.execute(first["id"])

    assert done["status"] == "executed" and done["result"]["simulated"] is True
    assert sum(done["result"]["clicks"].values()) > 0
    with pytest.raises(InvalidState):
        svc.execute(first["id"])

    second = svc.plan(GOAL, 1000)
    estimates = next(s["result"] for s in second["trace"] if s["tool"] == "estimate_variant_ctr")
    assert any(v["impressions_seen"] > 0 for v in estimates["variants"])
    history = next(s["result"] for s in second["trace"] if s["tool"] == "recall_past_runs")
    assert len(history["runs"]) == 1

    # The split moves toward the ads that earned clicks, so the learning loop is closed.
    assert max(second["plan"]["allocation"].values()) > max(first["plan"]["allocation"].values()) + 0.1
    assert second["grounding"]["ok"] and "impressions" in second["explanation"]
    actions = [e["action"] for e in list_events(settings.db_path)]
    assert {"agent_plan_proposed", "agent_plan_approved", "agent_plan_executed"} <= set(actions)


# ---- the LLM planner and its failure modes --------------------------------

def test_llm_plan_with_grounded_text_is_accepted(tmp_path, targeter):
    llm = ScriptedLLM(TOOL_STEP, honest)
    svc, _ = make_service(tmp_path, targeter, llm=llm)
    run = svc.plan(GOAL, 1000)

    assert run["planner"] == "llm" and run["grounding"]["explanation_source"] == "llm"
    assert run["grounding"]["ok"] and "%" in run["explanation"]
    assert "2026" not in run["plan"]["name"]  # digits are stripped from LLM-chosen names
    assert len(llm.calls[0]["tools"]) == 6


def test_an_invented_number_gets_one_rewrite_and_then_the_template(tmp_path, targeter):
    llm = ScriptedLLM(TOOL_STEP, BAD_TEXT, BAD_TEXT)
    svc, _ = make_service(tmp_path, targeter, llm=llm)
    run = svc.plan(GOAL, 1000)

    assert len(llm.calls) == 3
    assert "do not appear in any tool result" in llm.calls[2]["messages"][-1]["content"]
    assert run["grounding"]["explanation_source"] == "template" and run["grounding"]["ok"]
    assert run["grounding"]["llm_text_rejected"] is True
    assert "41.37%" in run["grounding"]["first_llm_text_rejected"]["ungrounded"]
    assert "41.37" not in run["explanation"]


def test_a_rewrite_can_repair_a_bad_first_answer(tmp_path, targeter):
    svc, _ = make_service(tmp_path, targeter, llm=ScriptedLLM(TOOL_STEP, BAD_TEXT, honest))
    run = svc.plan(GOAL, 1000)
    assert run["grounding"]["explanation_source"] == "llm" and run["grounding"]["ok"]
    assert "first_llm_text_rejected" in run["grounding"]


def test_a_false_claim_of_execution_is_rejected_even_without_numbers(tmp_path, targeter):
    boast = LLMResponse(content="Great news, I executed the campaign and your ads are now live.")
    llm = ScriptedLLM(TOOL_STEP, boast, boast)
    svc, _ = make_service(tmp_path, targeter, llm=llm)
    run = svc.plan(GOAL, 1000)

    assert "nothing has been run" in llm.calls[2]["messages"][-1]["content"]
    assert run["grounding"]["explanation_source"] == "template"
    assert run["grounding"]["first_llm_text_rejected"]["claims_action"] is True
    assert run["grounding"]["claims_action"] is False
    assert run["status"] == "proposed" and run["result"] is None


@pytest.mark.parametrize("bad_budget", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_non_finite_and_non_positive_budgets_are_refused(tmp_path, targeter, bad_budget):
    svc, _ = make_service(tmp_path, targeter)
    with pytest.raises(GuardrailViolation):
        svc.plan(GOAL, bad_budget)
    assert svc.list() == []


def test_an_llm_that_never_proposes_still_yields_a_plan(tmp_path, targeter):
    llm = ScriptedLLM(LLMResponse(content="Just trust me, 9999%."), LLMResponse(content="Still 9999%."))
    svc, _ = make_service(tmp_path, targeter, llm=llm)
    run = svc.plan(GOAL, 1000)
    assert run["planner"] == "llm_without_proposal" and run["plan"]["budget"] == 1000
    assert run["grounding"]["explanation_source"] == "template" and run["grounding"]["ok"]


def test_the_llm_cannot_propose_a_budget_different_from_the_owners(tmp_path, targeter):
    greedy = LLMResponse(tool_calls=[ToolCall("c1", "propose_plan", {"total_budget": 4000})])
    svc, _ = make_service(tmp_path, targeter, llm=ScriptedLLM(greedy, LLMResponse(content="Here is the plan.")))
    run = svc.plan(GOAL, 1000)

    refused = next(s for s in run["trace"] if s["args"].get("total_budget") == 4000)
    assert not refused["ok"] and "owner's budget" in refused["result"]["error"]
    assert run["plan"]["budget"] == 1000


def test_an_llm_failure_falls_back_to_the_offline_planner(tmp_path, targeter):
    svc, _ = make_service(tmp_path, targeter, llm=BrokenLLM())
    run = svc.plan(GOAL, 1000)
    assert run["planner"] == "rule_based_after_llm_error"
    assert "HTTP 500" in run["grounding"]["llm_error"]
    assert run["grounding"]["ok"]


def test_the_agent_has_no_tool_that_can_run_a_plan(tmp_path, targeter):
    llm = ScriptedLLM(TOOL_STEP, honest)
    svc, _ = make_service(tmp_path, targeter, llm=llm)
    svc.plan(GOAL, 1000)
    names = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert names == {"list_ad_variants", "estimate_variant_ctr", "recall_past_runs",
                     "recommend_budget_split", "rank_audience", "propose_plan"}


def test_every_stored_explanation_passes_the_grounding_check_again(tmp_path, targeter):
    svc, _ = make_service(tmp_path, targeter)
    run = svc.plan(GOAL, 1000)
    results = [s["result"] for s in run["trace"] if s["ok"]]
    assert check_grounding(run["explanation"], results, (GOAL, "1000")).ok


# ---- HTTP API --------------------------------------------------------------

@pytest.fixture
def client(tmp_path, targeter):
    settings = Settings(db_path=str(tmp_path / "api.db"), max_total_budget=5000.0, uplift_customers=4000)
    return TestClient(create_app(settings, targeter=targeter))


def test_api_full_flow_with_approval(client):
    assert client.get("/health").json()["planner"] == "rule_based"

    run = client.post("/agent/plan", json={"goal": GOAL, "budget": 1000}).json()
    run_id = run["id"]
    assert run["status"] == "proposed"

    assert client.post(f"/agent/runs/{run_id}/execute").status_code == 409  # not approved yet
    assert client.post(f"/agent/runs/{run_id}/approve", json={"approved": True}).json()["status"] == "approved"
    assert client.post(f"/agent/runs/{run_id}/approve", json={"approved": True}).status_code == 409

    done = client.post(f"/agent/runs/{run_id}/execute").json()
    assert done["status"] == "executed" and done["result"]["simulated"] is True
    assert client.post(f"/agent/runs/{run_id}/execute").status_code == 409
    assert client.get(f"/agent/runs/{run_id}").json()["status"] == "executed"
    assert len(client.get("/agent/runs").json()) == 1
    assert any(e["action"] == "agent_plan_executed" for e in client.get("/audit").json())


def test_api_state_endpoint_reports_settings_and_what_the_agent_has_learned(client):
    before = client.get("/agent/state").json()
    assert before["planner"] == "rule_based" and before["simulated"] is True
    assert before["max_total_budget"] == 5000.0 and before["exploration_floor"] == 0.01
    assert len(before["variants"]) == 5 and all(v["estimated_ctr"] is None for v in before["variants"])
    assert before["history"] == []

    run_id = client.post("/agent/plan", json={"goal": GOAL, "budget": 1000}).json()["id"]
    client.post(f"/agent/runs/{run_id}/approve", json={"approved": True})
    client.post(f"/agent/runs/{run_id}/execute")

    after = client.get("/agent/state").json()
    assert all(v["impressions_seen"] > 0 and 0 < v["ctr_low"] < v["estimated_ctr"] < v["ctr_high"] < 1
               for v in after["variants"])
    assert len(after["history"]) == 1 and after["history"][0]["run_id"] == run_id


def test_api_run_list_leaves_out_the_trace_unless_asked(client):
    client.post("/agent/plan", json={"goal": GOAL, "budget": 1000})
    light = client.get("/agent/runs").json()[0]
    full = client.get("/agent/runs?include_trace=true").json()[0]
    assert "trace" not in light and "plan" in light and "explanation" in light
    assert full["trace"][-1]["tool"] == "propose_plan"
    assert client.get("/agent/runs?limit=0").status_code == 422
    assert client.get("/audit?limit=100000").status_code == 422


def test_the_built_dashboard_is_served_at_the_root_without_shadowing_the_api(tmp_path, targeter, monkeypatch):
    import backend.main as main

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>VisionPilot</title><div id=root></div>")
    monkeypatch.setattr(main, "DASHBOARD_DIR", dist)
    settings = Settings(db_path=str(tmp_path / "dash.db"), uplift_customers=4000)
    served = TestClient(create_app(settings, targeter=targeter))

    assert "VisionPilot" in served.get("/").text
    assert served.get("/health").json()["status"] == "ok"   # API routes still win
    assert served.get("/agent/state").status_code == 200


def test_api_rejects_bad_requests(client):
    assert client.post("/agent/plan", json={"goal": GOAL, "budget": 99999}).status_code == 422
    assert client.post("/agent/plan", json={"goal": GOAL, "budget": -5}).status_code == 422
    assert client.post("/agent/plan", json={"goal": "", "budget": 100}).status_code == 422
    assert client.get("/agent/runs/12345").status_code == 404
    assert client.post("/agent/runs/12345/execute").status_code == 404
