import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pragmatic_rc import neural, optimal, random_estimation_mdp  # noqa: E402
from pragmatic_rc.envs import GridWorld, TabularSimulator  # noqa: E402


def test_learned_policies_land_within_the_exact_bounds_on_a_small_problem():
    mdp = random_estimation_mdp(3, 0.5, 0.8, np.random.default_rng(1))
    beta = 0.3
    lo, hi = optimal.solve(mdp, beta, eps=0.05).value_bounds
    sim = TabularSimulator(mdp, onehot=True)
    pol = neural.train(sim, beta, updates=250)
    mean, se, _ = neural.evaluate(sim, pol, beta)
    assert mean <= hi + 4 * se  # a learned policy cannot beat the optimum
    assert mean >= lo - 0.15  # and should be close to it


def test_batched_simulators_agree_with_the_scalar_ones():
    g = GridWorld(n=5, slip=0.2)
    rng = np.random.default_rng(0)
    s = np.full(20000, 7)
    s2, r = g.step_batch(s, np.full(20000, 4), rng)
    ref = np.bincount([g.step(7, 4, rng)[0] for _ in range(20000)], minlength=25) / 20000
    assert np.abs(np.bincount(s2, minlength=25) / 20000 - ref).max() < 0.02
