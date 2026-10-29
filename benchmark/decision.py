"""Decision-quality benchmark: does the learned core beat simple rules over repeated campaigns?

Each seed builds a different SYNTHETIC world (ads with different true click rates). Every
policy then runs several rounds of the same budget in it. The score is EXPECTED clicks, which
is the allocation's exact value under the world's true click rates, so there is no sampling
noise from the clicks themselves. Differences are paired by seed.

Policies
  equal_split  the same share for every ad, forever.
  greedy       one equal round, then all the budget on the best ad seen so far.
  agent        the real agent service (rule-based planner): Thompson sampling with the default
               minimum share per ad (1%), learning from each executed run.
  agent_floor_5pct  the same agent with the earlier 5% minimum share.
  agent_no_floor    the same agent with no minimum share. Together these show what the floor costs.
  oracle       all the budget on the truly best ad. Not achievable, only a ceiling.
  llm_alloc    (optional, needs your own LLM) the LLM picks shares from the results so far.
               This is the "LLM without the learned core" ablation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from backend.adapters.base import CampaignPlan
from backend.adapters.sandbox import SandboxAdapter
from backend.agent.service import AgentService
from backend.agent.targeting import UpliftTargeter
from backend.agent.tools import variant_names
from backend.config import Settings
from backend.db import init_db
from benchmark.stats import bootstrap_ci
from ml.simulator.bandit_env import AdEnvironment

COST_PER_IMPRESSION = 0.05  # the sandbox's price per impression


def true_arm_ctr(env: AdEnvironment, n: int = 100_000) -> np.ndarray:
    """Each ad's true average click rate in this world (the truth the policies cannot see)."""
    contexts = np.random.default_rng(12345).normal(size=(n, env.dim))
    return env.click_prob(contexts).mean(axis=0)


def expected_clicks(allocation: dict[int, float], budget: float, ctr: np.ndarray) -> float:
    return float(sum(budget * share / COST_PER_IMPRESSION * ctr[arm] for arm, share in allocation.items()))


def _equal(n_arms: int) -> dict[int, float]:
    return {a: 1.0 / n_arms for a in range(n_arms)}


class LLMAllocator:
    """Asks an LLM for budget shares given the results so far. Invalid answers return None."""

    SYSTEM = ("You allocate an ad budget across ads to maximise clicks. Reply with ONLY a JSON object "
              "mapping every ad name to a share between 0 and 1. The shares must sum to 1.")

    def __init__(self, llm, names: list[str]) -> None:
        self.llm, self.names = llm, names

    def allocate(self, impressions: np.ndarray, clicks: np.ndarray) -> dict[int, float] | None:
        table = "\n".join(f"{n}: {int(i)} impressions, {int(c)} clicks"
                          for n, i, c in zip(self.names, impressions, clicks))
        try:
            reply = self.llm.chat([{"role": "system", "content": self.SYSTEM},
                                   {"role": "user", "content": f"Results so far:\n{table}"}], tools=None)
            data = json.loads(re.search(r"\{.*\}", reply.content or "", re.DOTALL).group(0))
            shares = [float(data[n]) for n in self.names]
        except Exception:
            return None
        if set(data) != set(self.names) or any(not np.isfinite(s) or s < 0 for s in shares):
            return None
        total = sum(shares)
        if abs(total - 1.0) > 0.01:
            return None
        return {i: s / total for i, s in enumerate(shares)}


def _fixed_rounds(allocation: dict[int, float], rounds: int, budget: float, ctr: np.ndarray) -> list[float]:
    return [expected_clicks(allocation, budget, ctr)] * rounds


def _greedy_rounds(seed: int, rounds: int, budget: float, ctr: np.ndarray, world: dict) -> list[float]:
    env = AdEnvironment(seed=seed, **world)
    sandbox = SandboxAdapter(env, COST_PER_IMPRESSION)
    imps, clicks, out = np.zeros(env.n_arms), np.zeros(env.n_arms), []
    for r in range(rounds):
        if r == 0:
            alloc = _equal(env.n_arms)
        else:
            rate = np.where(imps > 0, clicks / np.maximum(imps, 1), -1.0)
            alloc = {int(rate.argmax()): 1.0}
        result = sandbox.execute(CampaignPlan("greedy", budget, alloc))
        for arm in alloc:
            imps[arm] += result.impressions[arm]
            clicks[arm] += result.clicks[arm]
        out.append(expected_clicks(alloc, budget, ctr))
    return out


def _llm_rounds(seed: int, rounds: int, budget: float, ctr: np.ndarray, llm, counters: dict,
                world: dict) -> list[float]:
    env = AdEnvironment(seed=seed, **world)
    sandbox = SandboxAdapter(env, COST_PER_IMPRESSION)
    allocator = LLMAllocator(llm, variant_names(env.n_arms))
    imps, clicks, out = np.zeros(env.n_arms), np.zeros(env.n_arms), []
    for r in range(rounds):
        alloc = _equal(env.n_arms)
        if r > 0:
            asked = allocator.allocate(imps, clicks)
            counters["asked"] += 1
            if asked is None:
                counters["invalid"] += 1  # unusable answer: fall back to an equal split
            else:
                alloc = asked
        result = sandbox.execute(CampaignPlan("llm", budget, alloc))
        for arm in alloc:
            imps[arm] += result.impressions[arm]
            clicks[arm] += result.clicks[arm]
        out.append(expected_clicks(alloc, budget, ctr))
    return out


def _agent_rounds(seed: int, rounds: int, budget: float, ctr: np.ndarray, targeter: UpliftTargeter,
                  workdir: str | Path, floor: float = Settings.exploration_floor, tag: str = "agent",
                  world: dict | None = None) -> list[float]:
    db = str(Path(workdir) / f"decision_{tag}_{budget:g}_seed{seed}.db")
    settings = Settings(db_path=db, max_total_budget=max(10_000.0, budget * 2),
                        uplift_customers=targeter.n_customers, exploration_floor=floor)
    init_db(db)
    env = AdEnvironment(seed=seed, **(world or {}))
    svc = AgentService(settings, SandboxAdapter(env, COST_PER_IMPRESSION), llm=None, targeter=targeter)
    out = []
    for _ in range(rounds):
        run = svc.plan("Get more customers", budget)
        svc.approve(run["id"], True)
        svc.execute(run["id"])
        alloc = {svc.names.index(name): share for name, share in run["plan"]["allocation"].items()}
        out.append(expected_clicks(alloc, budget, ctr))
    return out


def run_decision_benchmark(seeds: int, rounds: int, budget: float, targeter: UpliftTargeter,
                           workdir: str | Path, llm=None, world: dict | None = None,
                           world_name: str = "high_ctr") -> dict:
    world = world or {}
    per_round: dict[str, list[list[float]]] = {
        "equal_split": [], "greedy": [], "agent": [], "agent_floor_5pct": [], "agent_no_floor": [], "oracle": [],
    }
    counters = {"asked": 0, "invalid": 0}
    if llm is not None:
        per_round["llm_alloc"] = []

    for seed in range(seeds):
        env = AdEnvironment(seed=seed, **world)
        ctr = true_arm_ctr(env)
        per_round["equal_split"].append(_fixed_rounds(_equal(env.n_arms), rounds, budget, ctr))
        per_round["oracle"].append(_fixed_rounds({int(ctr.argmax()): 1.0}, rounds, budget, ctr))
        per_round["greedy"].append(_greedy_rounds(seed, rounds, budget, ctr, world))
        per_round["agent"].append(_agent_rounds(seed, rounds, budget, ctr, targeter, workdir, tag=f"agent_{world_name}", world=world))
        per_round["agent_floor_5pct"].append(
            _agent_rounds(seed, rounds, budget, ctr, targeter, workdir, floor=0.05, tag=f"floor5_{world_name}", world=world))
        per_round["agent_no_floor"].append(
            _agent_rounds(seed, rounds, budget, ctr, targeter, workdir, floor=0.0, tag=f"nofloor_{world_name}", world=world))
        if llm is not None:
            per_round["llm_alloc"].append(_llm_rounds(seed, rounds, budget, ctr, llm, counters, world))

    totals = {name: np.array([sum(r) for r in rows]) for name, rows in per_round.items()}
    oracle, equal, greedy = totals["oracle"], totals["equal_split"], totals["greedy"]
    policies = {}
    for name, arr in totals.items():
        mean, lo, hi = bootstrap_ci(arr)
        entry = {
            "total_expected_clicks": {"mean": mean, "ci95": [lo, hi]},
            "share_of_oracle": float(arr.mean() / oracle.mean()),
            "per_round_mean": np.mean(per_round[name], axis=0).tolist(),
            "regret_vs_oracle": dict(zip(("mean", "ci95_lo", "ci95_hi"), bootstrap_ci(oracle - arr))),
        }
        for reference_name, reference in (("equal_split", equal), ("greedy", greedy)):
            if name != reference_name:
                d_mean, d_lo, d_hi = bootstrap_ci(arr - reference)
                entry[f"gain_over_{reference_name}"] = {"mean": d_mean, "ci95": [d_lo, d_hi],
                                                        "seeds_better": float((arr > reference).mean())}
        policies[name] = entry
    return {
        "synthetic": True,
        "config": {"seeds": seeds, "rounds": rounds, "budget_per_round": budget,
                   "cost_per_impression": COST_PER_IMPRESSION, "world": world_name, "world_params": world,
                   "mean_arm_ctr": float(np.mean([true_arm_ctr(AdEnvironment(seed=s, **world)).mean() for s in range(seeds)]))},
        "policies": policies,
        "llm_invalid_allocations": counters if llm is not None else None,
    }
