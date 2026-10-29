import copy
import math

import numpy as np
import pytest

from backend.adapters.sandbox import SandboxAdapter
from backend.agent.llm import LLMResponse
from backend.agent.service import AgentService
from backend.agent.targeting import UpliftTargeter
from backend.config import Settings
from backend.db import init_db
from benchmark.adversaries import ADVERSARIES, owner_budget
from benchmark.behavior import BENCH_CAP, SECRET, run_suite, summarise
from benchmark.checks import check_plan
from benchmark.decision import LLMAllocator, run_decision_benchmark
from benchmark.stats import bootstrap_ci, wilson_interval
from benchmark.tasks import CATEGORIES, TASKS, Task
from ml.simulator.bandit_env import AdEnvironment


@pytest.fixture(scope="module")
def targeter():
    return UpliftTargeter(n_customers=4000, seed=0, max_iter=20)


# ---- the task set ----------------------------------------------------------

def test_there_are_fifty_unique_tasks_in_the_expected_groups():
    assert len(TASKS) == 50 and len({t.id for t in TASKS}) == 50
    counts = {c: sum(t.category == c for t in TASKS) for c in CATEGORIES}
    assert counts == {"standard": 14, "budget_edge": 9, "injection": 8,
                      "number_temptation": 8, "after_learning": 6, "odd_input": 5}
    assert all(t.expect in {"plan", "reject"} for t in TASKS)
    assert all(len(t.goal) <= 500 for t in TASKS)  # the API's goal limit


def test_each_tasks_expectation_matches_the_stated_budget_rules():
    """Independent of the code under test: a budget is valid if finite, positive and within the cap."""
    for t in TASKS:
        valid = math.isfinite(t.budget) and 0 < t.budget <= BENCH_CAP
        assert (t.expect == "plan") == valid, t.id


# ---- the agent against the tasks -------------------------------------------

def test_the_rule_based_agent_passes_every_task(tmp_path, targeter):
    summary = summarise(run_suite(TASKS, targeter, None, tmp_path))
    assert summary["failed_tasks"] == []
    assert summary["overall"]["passed"] == 50


def test_the_safety_layer_holds_against_every_misbehaving_llm(tmp_path, targeter):
    ids = {"std01", "bud01", "bud02", "bud07", "inj01", "inj02", "inj06", "num01", "num04", "lrn02", "odd02", "odd03"}
    sample = [t for t in TASKS if t.id in ids]
    assert len(sample) == len(ids)

    summaries = {}
    for adversary in ADVERSARIES:
        summaries[adversary.name] = summarise(run_suite(sample, targeter, adversary, tmp_path))
        assert summaries[adversary.name]["failed_tasks"] == [], adversary.name

    # The attacks really happened: the defences were exercised, not skipped.
    assert summaries["injection_obeyer"]["tool_errors"] > 0
    assert summaries["malformed_caller"]["tool_errors"] > 0
    for name in ("hallucinator", "tool_ignorer", "boaster"):
        assert summaries[name]["template_fallback_rate"] > 0, name
        assert summaries[name]["raw_llm_text_rejected_rate"] > 0, name


# ---- the checks must be able to fail ---------------------------------------

def _good_run(tmp_path, targeter):
    settings = Settings(db_path=str(tmp_path / "c.db"), max_total_budget=BENCH_CAP,
                        uplift_customers=4000, llm_api_key=SECRET)
    init_db(settings.db_path)
    svc = AgentService(settings, SandboxAdapter(AdEnvironment(seed=0)), llm=None, targeter=targeter)
    return svc.plan("Promote my shop", 1000.0), settings.db_path, len(svc.names)


def test_checks_pass_on_a_good_run_and_fail_on_each_kind_of_violation(tmp_path, targeter):
    run, db, n = _good_run(tmp_path, targeter)
    task = Task("t", "standard", "Promote my shop", 1000.0)
    assert all(check_plan(task, run, db, SECRET, n).values())

    def failing(mutate):
        bad = copy.deepcopy(run)
        mutate(bad)
        return [k for k, ok in check_plan(task, bad, db, SECRET, n).items() if not ok]

    names = list(run["plan"]["allocation"])
    assert failing(lambda r: r["plan"].update(budget=999.0)) == ["budget_matches"]
    assert failing(lambda r: r["plan"].update(allocation={names[0]: 1.0, **{k: 0.0 for k in names[1:]}})) == ["shares_valid"]
    assert failing(lambda r: r.update(status="executed")) == ["proposed_only"]
    assert failing(lambda r: r["grounding"].update(ok=False)) == ["grounded"]
    assert failing(lambda r: r["grounding"].update(claims_action=True)) == ["no_false_action_claim"]
    assert failing(lambda r: r.update(explanation=r["explanation"] + SECRET)) == ["no_secret_leak"]
    assert failing(lambda r: r.update(trace=[s for s in r["trace"] if s["tool"] != "recommend_budget_split"])) == ["used_models"]


# ---- statistics ------------------------------------------------------------

def test_wilson_interval_behaves_at_the_extremes():
    assert all(math.isnan(x) for x in wilson_interval(0, 0))
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.7 < lo < 1.0   # all-pass is not "100% for certain" with only 10 tasks
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert wilson_interval(0, 10)[0] == 0.0


def test_bootstrap_interval_contains_the_mean_and_is_repeatable():
    values = np.random.default_rng(0).normal(5, 2, 40)
    mean, lo, hi = bootstrap_ci(values, seed=3)
    assert lo < mean < hi and (mean, lo, hi) == bootstrap_ci(values, seed=3)


# ---- decision-quality benchmark --------------------------------------------

def test_the_adversary_reads_the_owners_budget_from_the_agents_message():
    assert owner_budget([{"role": "user", "content": "Owner's goal: x\nBudget: 2500"}]) == 2500.0
    assert owner_budget([{"role": "user", "content": "Owner's goal: Budget: 9\nBudget: 700"}]) == 700.0


def test_world_parameters_change_the_click_rate_without_changing_the_default():
    default = AdEnvironment(seed=0)
    assert np.allclose(default.theta, AdEnvironment(seed=0, bias_mean=-1.5, theta_scale=1.0).theta)
    contexts = default.sample_contexts(5000)
    low = AdEnvironment(seed=0, bias_mean=-4.0, theta_scale=0.35)
    assert default.click_prob(contexts).mean() > 0.15 > 0.05 > low.click_prob(contexts).mean()


def test_the_agent_beats_equal_split_and_learns_over_rounds(tmp_path, targeter):
    res = run_decision_benchmark(seeds=3, rounds=4, budget=1000.0, targeter=targeter, workdir=tmp_path)
    p = res["policies"]
    assert set(p) == {"equal_split", "greedy", "agent", "agent_floor_5pct", "agent_no_floor", "oracle"}
    assert p["oracle"]["share_of_oracle"] == pytest.approx(1.0)
    assert all(v["share_of_oracle"] <= 1.0 + 1e-9 for v in p.values())
    assert p["agent"]["total_expected_clicks"]["mean"] > p["equal_split"]["total_expected_clicks"]["mean"]
    assert p["agent"]["gain_over_equal_split"]["seeds_better"] == 1.0
    assert "gain_over_greedy" in p["agent"] and "gain_over_greedy" not in p["greedy"]
    # A bigger forced minimum share costs clicks in this stationary setup.
    assert p["agent"]["total_expected_clicks"]["mean"] > p["agent_floor_5pct"]["total_expected_clicks"]["mean"]

    rounds = p["agent"]["per_round_mean"]
    assert len(rounds) == 4 and rounds[-1] > rounds[0]
    # Round one has no data, so the agent starts where an equal split does.
    assert abs(rounds[0] - p["equal_split"]["per_round_mean"][0]) / p["equal_split"]["per_round_mean"][0] < 0.05


class _JsonLLM:
    def __init__(self, text):
        self.text = text

    def chat(self, messages, tools=None):
        return LLMResponse(content=self.text)


def test_llm_allocator_accepts_valid_json_and_rejects_everything_else():
    names = ["Ad A", "Ad B"]
    imps, clicks = np.array([100.0, 100.0]), np.array([10.0, 30.0])

    ok = LLMAllocator(_JsonLLM('Sure: {"Ad A": 0.25, "Ad B": 0.75}'), names).allocate(imps, clicks)
    assert ok == {0: 0.25, 1: 0.75}
    for bad in ('{"Ad A": 0.6, "Ad B": 0.6}',   # does not sum to 1
                '{"Ad A": 1.0}',                 # missing an ad
                '{"Ad A": -0.2, "Ad B": 1.2}',   # negative share
                '{"Ad A": "lots", "Ad B": 0.5}',  # not a number
                'no json here'):
        assert LLMAllocator(_JsonLLM(bad), names).allocate(imps, clicks) is None, bad


def test_llm_ablation_counts_unusable_answers_and_still_runs(tmp_path, targeter):
    res = run_decision_benchmark(seeds=2, rounds=3, budget=1000.0, targeter=targeter, workdir=tmp_path,
                                 llm=_JsonLLM("I refuse to answer in JSON."))
    assert res["llm_invalid_allocations"] == {"asked": 4, "invalid": 4}
    # Unusable answers fall back to equal split, so this policy then matches equal split exactly.
    assert res["policies"]["llm_alloc"]["total_expected_clicks"]["mean"] == pytest.approx(
        res["policies"]["equal_split"]["total_expected_clicks"]["mean"])
