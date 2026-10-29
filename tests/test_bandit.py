import numpy as np

from ml.bandit.online import run_online
from ml.bandit.ope import estimate_policy_value
from ml.bandit.policies import BernoulliThompson, LinUCB, LogisticGreedy, UniformRandom
from ml.simulator.bandit_env import AdEnvironment

N_ARMS, DIM = 5, 6


def test_click_probabilities_are_valid():
    env = AdEnvironment(N_ARMS, DIM, seed=0)
    probs = env.click_prob(env.sample_contexts(1000))
    assert probs.shape == (1000, N_ARMS)
    assert ((probs > 0) & (probs < 1)).all()


def test_oracle_beats_uniform():
    env = AdEnvironment(N_ARMS, DIM, seed=0)
    assert env.oracle_value() > env.policy_value(UniformRandom(N_ARMS))


def test_logged_propensities_match_the_logging_policy():
    env = AdEnvironment(N_ARMS, DIM, seed=0)
    log = env.log(UniformRandom(N_ARMS), 2000)
    assert np.allclose(log.propensities, 1.0 / N_ARMS)
    assert set(np.unique(log.actions)) <= set(range(N_ARMS))


def _regret(policy, seed=1, rounds=3000):
    env = AdEnvironment(N_ARMS, DIM, seed=seed)
    return run_online(env, policy, rounds).cumulative_regret[-1]


def test_learning_policies_lose_fewer_clicks_than_random():
    random_regret = _regret(UniformRandom(N_ARMS, seed=1))
    assert _regret(LinUCB(N_ARMS, DIM, seed=1)) < random_regret
    assert _regret(BernoulliThompson(N_ARMS, seed=1)) < random_regret


def test_off_policy_estimates_are_close_to_the_true_value():
    env = AdEnvironment(N_ARMS, DIM, seed=2)
    logging_policy = UniformRandom(N_ARMS, seed=2)
    target = LogisticGreedy(N_ARMS).fit(env.log(logging_policy, 5000))
    logged = env.log(logging_policy, 60_000)

    truth = env.policy_value(target)
    est = estimate_policy_value(logged, target, seed=2)

    assert abs(est.ipw - truth) < 0.02
    assert abs(est.snipw - truth) < 0.02
    assert abs(est.dr - truth) < 0.02
    assert est.effective_sample_size > 1000
