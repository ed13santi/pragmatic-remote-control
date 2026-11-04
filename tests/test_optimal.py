import numpy as np
import pytest

from pragmatic_rc import optimal, pull, push, random_control_mdp, random_estimation_mdp


def mdp_pair(seed=1, n_states=4):
    rng = np.random.default_rng(seed)
    return {
        "control": random_control_mdp(n_states, 2, 0.5, 0.9, rng),
        "estimation": random_estimation_mdp(n_states, 0.5, 0.9, rng),
    }


@pytest.mark.parametrize("task", ["control", "estimation"])
@pytest.mark.parametrize("beta", [0.1, 0.4])
def test_bounds_are_ordered_and_tight(task, beta):
    sol = optimal.solve(mdp_pair()[task], beta, eps=0.02)
    assert np.all(sol.lower <= sol.upper + 1e-9)
    assert np.all(sol.upper - sol.lower <= 0.02 + 1e-9)


@pytest.mark.parametrize("task", ["control", "estimation"])
@pytest.mark.parametrize("beta", [0.1, 0.4])
def test_optimal_dominates_alternating_and_pull(task, beta):
    mdp, t_max = mdp_pair()[task], 8
    _, upper = optimal.solve(mdp, beta, eps=0.02).value_bounds
    n = mdp.n_states
    api = max(push.api(mdp, beta, t_max, c0)[1].value(beta).mean() for c0 in (push.never_transmit(t_max, n), push.always_transmit(t_max, n)))
    mpi = pull.mpi(mdp, beta, t_max, restarts=10)[1].value(beta).mean()
    assert upper >= api - 1e-9
    assert upper >= mpi - 1e-9


@pytest.mark.parametrize("task", ["control", "estimation"])
def test_executed_policy_matches_the_bounds(task):
    mdp, beta = mdp_pair(2, n_states=3)[task], 0.3
    sol = optimal.solve(mdp, beta, eps=0.02)
    lo, hi = sol.value_bounds
    r, tx, r_se, tx_se = optimal.simulate(sol, 1500, np.random.default_rng(0))
    value, se = r - beta * tx, np.hypot(r_se, beta * tx_se)
    assert lo - 4 * se <= value <= hi + 4 * se


@pytest.mark.parametrize("task", ["control", "estimation"])
def test_free_channel_recovers_full_information_value(task):
    mdp = mdp_pair()[task]
    sol = optimal.solve(mdp, 0.0, eps=0.02)
    v = np.zeros(mdp.n_states)
    for _ in range(2000):
        v = (mdp.Rbar + mdp.gamma * mdp.P @ v).max(axis=0)
    assert sol.lower.mean() - 1e-9 <= v.mean() <= sol.upper.mean() + 1e-9
