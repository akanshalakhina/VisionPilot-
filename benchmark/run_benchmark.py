"""Run the whole benchmark.

Usage:
  python -m benchmark.run_benchmark --out reports/benchmark.json
  python -m benchmark.run_benchmark --llm        # also runs YOUR configured LLM (see .env / docs/agent.md)

Parts
  A1  offline rule-based agent on all 50 tasks.
  A2  the safety layer against 6 deliberately misbehaving scripted LLMs (not a test of real LLMs).
  A3  (--llm) your real LLM planner on all 50 tasks.
  B   decision quality over repeated campaigns in synthetic worlds, against simple baselines.
  B2  (--llm) the LLM choosing budget shares by itself, with no learned core.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from backend.agent.llm import build_llm
from backend.agent.targeting import UpliftTargeter
from backend.config import Settings
from benchmark.adversaries import ADVERSARIES
from benchmark.behavior import run_suite, summarise
from benchmark.decision import run_decision_benchmark
from benchmark.tasks import TASKS


WORLDS = {
    # Clicks are plentiful and the ads differ a lot (about a 25% click rate). Easy for any adaptive rule.
    "high_ctr": {},
    # About a 2% click rate with small gaps between ads, closer to real advertising. Early results are
    # noisy here, so committing to one ad too soon is risky.
    "low_ctr": {"bias_mean": -4.0, "theta_scale": 0.35},
}


def _fmt_rate(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{block['passed']:>2}/{block['n']:<2} = {block['rate']:>6.1%}  (95% CI {lo:.1%} to {hi:.1%})"


def print_behavior(title: str, summary: dict) -> None:
    print(f"\n{title}")
    print(f"  overall           {_fmt_rate(summary['overall'])}")
    for category, block in summary["by_category"].items():
        print(f"  {category:<17} {_fmt_rate(block)}")
    if summary["check_failures"]:
        print(f"  failed checks: {summary['check_failures']}")
    if summary["failed_tasks"]:
        print("  failed tasks: " + ", ".join(f"{f['task']} {f['failed_checks'] or f['crashed']}" for f in summary["failed_tasks"][:12]))
    if summary["llm_runs"]:
        print(f"  first LLM text rejected by the checks: {summary['raw_llm_text_rejected_rate']:.1%}   "
              f"fell back to the template: {summary['template_fallback_rate']:.1%}   tool errors: {summary['tool_errors']}")


def _gain(p: dict, reference: str) -> str:
    g = p.get(f"gain_over_{reference}")
    if g is None:
        return ""
    return f"{g['mean']:>+8,.0f} [{g['ci95'][0]:+,.0f}, {g['ci95'][1]:+,.0f}] {g['seeds_better']:>4.0%}"


def print_decision(res: dict) -> None:
    cfg = res["config"]
    per_round = cfg["budget_per_round"] / cfg["cost_per_impression"]
    print(f"\nB  Decision quality, world '{cfg['world']}' (average click rate {cfg['mean_arm_ctr']:.1%}): "
          f"{cfg['seeds']} synthetic worlds x {cfg['rounds']} rounds, "
          f"budget {cfg['budget_per_round']:g} per round (about {per_round:,.0f} impressions per round)")
    print(f"  {'policy':<15} {'expected clicks':>10} {'% oracle':>9}   {'gain over equal split [95% CI] seeds better':<46}   gain over greedy")
    for name, p in res["policies"].items():
        m = p["total_expected_clicks"]
        print(f"  {name:<15} {m['mean']:>10,.0f} {p['share_of_oracle']:>9.1%}   "
              f"{_gain(p, 'equal_split'):<46}   {_gain(p, 'greedy')}")
    if res["llm_invalid_allocations"]:
        c = res["llm_invalid_allocations"]
        print(f"  LLM allocations that were unusable: {c['invalid']} of {c['asked']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--budgets", type=float, nargs="+", default=[20.0, 100.0, 1000.0],
                        help="budget per round; small budgets mean little data per round, which is where exploration matters")
    parser.add_argument("--uplift-customers", type=int, default=8000)
    parser.add_argument("--worlds", nargs="+", default=list(WORLDS), choices=list(WORLDS),
                        help="which synthetic world types to run")
    parser.add_argument("--llm", action="store_true", help="also run the LLM configured in your environment")
    parser.add_argument("--out", default="reports/benchmark.json")
    args = parser.parse_args()

    llm = None
    if args.llm:
        llm = build_llm(Settings.from_env())
        if llm is None:
            parser.error("--llm needs LLM_PROVIDER, LLM_MODEL and LLM_API_KEY in your environment (your own key)")

    targeter = UpliftTargeter(args.uplift_customers, seed=0)
    report: dict = {"tasks": len(TASKS), "synthetic": True}
    print("SYNTHETIC WORLDS - clicks, customers and audiences are simulated, not real campaign data")

    with tempfile.TemporaryDirectory() as work:
        report["A1_rule_based"] = summarise(run_suite(TASKS, targeter, None, work))
        print_behavior("A1  Offline rule-based agent on 50 tasks", report["A1_rule_based"])

        report["A2_safety_layer_vs_adversarial_llms"] = {}
        print("\nA2  Safety layer against misbehaving scripted LLMs (50 tasks each)")
        for adversary in ADVERSARIES:
            summary = summarise(run_suite(TASKS, targeter, adversary, work))
            report["A2_safety_layer_vs_adversarial_llms"][adversary.name] = summary
            print(f"  {adversary.name:<17} {_fmt_rate(summary['overall'])}  tool errors {summary['tool_errors']}")
            if summary["failed_tasks"]:
                print("    failures: " + ", ".join(f"{f['task']} {f['failed_checks'] or f['crashed']}" for f in summary["failed_tasks"][:8]))

        if llm is not None:
            report["A3_real_llm"] = summarise(run_suite(TASKS, targeter, lambda: llm, work))
            print_behavior(f"A3  Your LLM planner ({llm.model}) on 50 tasks", report["A3_real_llm"])

        report["B_decision_quality"] = {}
        for world_name in args.worlds:
            for budget in args.budgets:
                result = run_decision_benchmark(args.seeds, args.rounds, budget, targeter, work, llm,
                                                world=WORLDS[world_name], world_name=world_name)
                report["B_decision_quality"][f"{world_name}/{budget:g}"] = result
                print_decision(result)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
