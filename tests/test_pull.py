import numpy as np
import pytest

from pragmatic_rc import pull, random_control_mdp
from pragmatic_rc.simulate import simulate_pull


def small_mdp(seed=0, n_states=4, n_actions=2, density=0.5, gamma=0.9):
    return random_control_mdp(n_states, n_actions, density, gamma, np.random.default_rng(seed))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5])
def test_mpi_matches_brute_force(seed, beta):
    mdp, t_max = small_mdp(seed), 4
    _, ev = pull.mpi(mdp, beta, t_max, restarts=40, rng=np.random.default_rng(seed))
    optimum = pull.brute_force_value(mdp, beta, t_max)
    np.testing.assert_allclose(ev.value(beta), optimum, atol=1e-8)


def test_evaluation_matches_simulation():
    mdp, beta, t_max = small_mdp(3, n_states=6), 0.2, 5
    rng = np.random.default_rng(1)
    policy = pull.random_policy(mdp, t_max, rng)
    ev = pull.evaluate(mdp, policy, t_max)
    sim = simulate_pull(mdp, policy, 40_000, rng)
    assert abs(sim.reward - ev.reward.mean()) < 4 * sim.reward_se
    assert abs(sim.tx - ev.tx.mean()) < 4 * sim.tx_se


def test_periodic_never_beats_adaptive():
    mdp, beta, t_max = small_mdp(2, n_states=8, density=0.3), 0.15, 6
    _, adaptive = pull.mpi(mdp, beta, t_max, restarts=20)
    for period in range(1, t_max + 1):
        _, periodic = pull.mpi(mdp, beta, t_max, fixed_tau=period)
        assert periodic.value(beta).mean() <= adaptive.value(beta).mean() + 1e-9


def test_single_start_improves_on_initial_policy_and_terminates():
    mdp, beta, t_max = small_mdp(2), 0.1, 4
    rng = np.random.default_rng(5)
    for _ in range(10):
        init = pull.random_policy(mdp, t_max, rng)
        before = pull.evaluate(mdp, init, t_max).value(beta)
        _, after = pull.mpi(mdp, beta, t_max, init=init)
        assert np.all(after.value(beta) >= before - 1e-9)


def test_free_channel_means_transmit_every_step():
    mdp, t_max = small_mdp(4), 4
    policy, ev = pull.mpi(mdp, 0.0, t_max)
    # With no cost, the decoder is never worse off by observing the state.
    full_info = pull.evaluate(mdp, pull.PullPolicy(np.ones(mdp.n_states, int), policy.pi_d), t_max)
    np.testing.assert_allclose(ev.value(0.0), full_info.value(0.0), atol=1e-8)
