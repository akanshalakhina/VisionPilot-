"""Bandit + off-policy-evaluation experiment on the SYNTHETIC ad environment.

Usage:  python -m ml.bandit.experiment --seed 0 --out reports/bandit_demo.json

Two questions are answered:
  1. Online: how fast do learning policies stop losing clicks compared with random?
  2. Offline: can we judge a policy from a log collected by a different policy?
     The environment knows the true value of every policy, so each estimator's
     error is measured directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.bandit.online import run_online
from ml.bandit.ope import estimate_policy_value
from ml.bandit.policies import BernoulliThompson, LinUCB, LogisticGreedy, Policy, UniformRandom, one_hot
from ml.simulator.bandit_env import AdEnvironment


class OraclePolicy(Policy):
    """Always shows the truly best ad. Only for reference; a real agent cannot do this."""

    def __init__(self, env: AdEnvironment) -> None:
        super().__init__(env.n_arms)
        self.env = env

    def action_probs(self, contexts: np.ndarray) -> np.ndarray:
        return one_hot(self.env.click_prob(contexts).argmax(axis=1), self.n_arms)


def run_demo(seed: int = 0, n_arms: int = 5, dim: int = 6, n_train: int = 5000, n_logged: int = 50_000) -> dict:
    env = AdEnvironment(n_arms, dim, seed)

    learners = {
        "uniform": UniformRandom(n_arms, seed),
        "thompson": BernoulliThompson(n_arms, seed),
        "linucb": LinUCB(n_arms, dim, alpha=1.0, seed=seed),
    }
    online = {}
    for name, policy in learners.items():
        result = run_online(env, policy, n_train)
        online[name] = {
            "cumulative_regret": float(result.cumulative_regret[-1]),
            "ctr_last_1000": float(result.rewards[-1000:].mean()),
        }

    logging_policy = UniformRandom(n_arms, seed)
    logistic = LogisticGreedy(n_arms).fit(env.log(logging_policy, n_train))
    eval_log = env.log(logging_policy, n_logged)

    candidates = {
        "uniform": logging_policy,
        "thompson (trained)": learners["thompson"],
        "linucb (trained)": learners["linucb"],
        "logistic greedy": logistic,
        "oracle (reference)": OraclePolicy(env),
    }
    offline = {}
    for name, policy in candidates.items():
        truth = env.policy_value(policy)
        est = estimate_policy_value(eval_log, policy, seed=seed)
        offline[name] = {
            "true_value": truth,
            "estimates": est.as_dict(),
            "abs_error": {k: abs(getattr(est, k) - truth) for k in ("ipw", "snipw", "dm", "dr")},
        }

    return {
        "synthetic": True,
        "config": {"seed": seed, "n_arms": n_arms, "dim": dim, "n_train": n_train, "n_logged": n_logged},
        "oracle_value": env.oracle_value(),
        "online": online,
        "offline": offline,
    }


def print_report(res: dict) -> None:
    print("SYNTHETIC ENVIRONMENT - results are simulated, not real campaign data\n")
    print(f"Oracle click rate (best ad every time): {res['oracle_value']:.4f}\n")
    print("Online learning (lower regret is better)")
    print(f"  {'policy':<10} {'cum. regret':>12} {'CTR last 1000':>14}")
    for name, row in res["online"].items():
        print(f"  {name:<10} {row['cumulative_regret']:>12.1f} {row['ctr_last_1000']:>14.4f}")
    print("\nOff-policy evaluation from a uniform-random log (absolute error vs true value)")
    print(f"  {'policy':<20} {'true':>7} {'IPW':>8} {'SNIPW':>8} {'DM':>8} {'DR':>8} {'ESS':>8}")
    for name, row in res["offline"].items():
        err, est = row["abs_error"], row["estimates"]
        print(
            f"  {name:<20} {row['true_value']:>7.4f} {err['ipw']:>8.4f} {err['snipw']:>8.4f} "
            f"{err['dm']:>8.4f} {err['dr']:>8.4f} {est['effective_sample_size']:>8.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/bandit_demo.json")
    args = parser.parse_args()

    res = run_demo(seed=args.seed)
    print_report(res)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
