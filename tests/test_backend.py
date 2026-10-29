import pytest
from fastapi.testclient import TestClient

from backend.adapters.base import AdapterDisabledError, CampaignPlan
from backend.adapters.meta import MetaAdapter
from backend.config import Settings
from backend.main import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(db_path=str(tmp_path / "test.db"), max_total_budget=1000.0)
    return TestClient(create_app(settings))


def _plan(**overrides):
    body = {"name": "diwali", "budget": 100.0, "allocation": {"0": 0.5, "1": 0.5}, "approved": True}
    body.update(overrides)
    return body


def test_health_reports_meta_adapter_off(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["meta_adapter_enabled"] is False


def test_unapproved_plan_is_rejected_and_logged(client):
    res = client.post("/sandbox/run", json=_plan(approved=False))
    assert res.status_code == 409
    events = client.get("/audit").json()
    assert events[0]["action"] == "plan_rejected"


def test_approved_plan_runs_in_the_simulator_and_is_logged(client):
    res = client.post("/sandbox/run", json=_plan())
    assert res.status_code == 200
    body = res.json()
    assert body["simulated"] is True
    assert body["adapter"] == "sandbox"
    assert body["impressions"]["0"] == 1000  # 100 * 0.5 / 0.05
    assert 0 <= body["clicks"]["0"] <= body["impressions"]["0"]
    assert client.get("/audit").json()[0]["action"] == "plan_executed"


def test_budget_over_the_cap_is_rejected(client):
    res = client.post("/sandbox/run", json=_plan(budget=5000.0))
    assert res.status_code == 422


def test_allocation_must_sum_to_one(client):
    res = client.post("/sandbox/run", json=_plan(allocation={"0": 0.5, "1": 0.2}))
    assert res.status_code == 422


def test_meta_adapter_refuses_to_run_when_disabled():
    plan = CampaignPlan(name="x", budget=10.0, allocation={0: 1.0})
    with pytest.raises(AdapterDisabledError):
        MetaAdapter(enabled=False).execute(plan)
