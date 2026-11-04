import numpy as np

from pragmatic_rc import optimal, random_estimation_mdp
from pragmatic_rc.envs import GridWorld, TabularSimulator
from pragmatic_rc.pomcp import POMCP, AllSubsets, DistanceFromMean, run_episode


def net_value(agent, beta, episodes, horizon, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        e = run_episode(agent, horizon, rng)
        out.append(e.reward - beta * e.transmissions)
    return np.mean(out), np.std(out) / np.sqrt(episodes)


def test_pomcp_matches_exact_bounds_on_a_small_estimation_problem():
    mdp = random_estimation_mdp(3, 0.5, 0.8, np.random.default_rng(1))
    beta = 0.3
    lo, hi = optimal.solve(mdp, beta, eps=0.05).value_bounds
    agent = POMCP(TabularSimulator(mdp), AllSubsets(3), beta, n_sims=150, seed=0)
    mean, se = net_value(agent, beta, episodes=15, horizon=25, seed=5)
    assert mean <= hi + 4 * se  # cannot beat the optimum
    assert mean >= lo - 0.3  # and gets close to it


def test_gridworld_tensors_match_sampler():
    g = GridWorld(n=4, slip=0.2)
    mdp = g.to_mdp()
    assert np.allclose(mdp.P.sum(-1), 1)
    rng = np.random.default_rng(0)
    s, a = 5, 2
    draws = np.bincount([g.step(s, a, rng)[0] for _ in range(20000)], minlength=16) / 20000
    assert np.abs(draws - mdp.P[a, s]).max() < 0.02


def test_free_runs_on_a_grid_larger_than_exact_solvers_can_handle():
    g = GridWorld(n=8)
    agent = POMCP(g, DistanceFromMean([-1, 0, 1, 2, float("inf")]), 0.05, n_sims=30, seed=0)
    e = run_episode(agent, 8, np.random.default_rng(0))
    assert 0 <= e.sends <= 8
