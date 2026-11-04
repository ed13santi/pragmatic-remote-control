import numpy as np
import pytest

from pragmatic_rc import pull, push, random_control_mdp, random_estimation_mdp
from pragmatic_rc.simulate import simulate_push


def control_mdp(seed=0, n_states=6, n_actions=2, density=0.4, gamma=0.9):
    return random_control_mdp(n_states, n_actions, density, gamma, np.random.default_rng(seed))


def test_evaluation_matches_simulation_with_informative_silence():
    mdp, t_max = control_mdp(1), 4
    rng = np.random.default_rng(2)
    c = rng.integers(0, 2, size=(t_max + 1, mdp.n_states, mdp.n_states))
    c[0] = 0
    c[t_max] = 1
    policy = push.PushPolicy(rng.integers(0, mdp.n_actions, size=(t_max, mdp.n_states)), c)
    ev = push.evaluate(mdp, policy, t_max)
    sim = simulate_push(mdp, policy, 60_000, rng)
    assert abs(sim.reward - ev.reward.mean()) < 4 * sim.reward_se
    assert abs(sim.tx - ev.tx.mean()) < 4 * sim.tx_se


@pytest.mark.parametrize("init", ["never", "always"])
def test_api_reaches_a_nash_equilibrium(init):
    mdp, beta, t_max = control_mdp(3), 0.2, 4
    c0 = (push.never_transmit if init == "never" else push.always_transmit)(t_max, mdp.n_states)
    policy, ev = push.api(mdp, beta, t_max, c0)
    # Neither player can improve by deviating unilaterally.
    c_dev = push.encoder_best_response(mdp, policy.pi_d, policy.c, beta, t_max)
    assert push.evaluate(mdp, push.PushPolicy(policy.pi_d, c_dev), t_max).value(beta).mean() <= ev.value(beta).mean() + 1e-8
    pi_dev = push.decoder_best_response(mdp, policy.pi_d, policy.c, beta, t_max)
    assert push.evaluate(mdp, push.PushPolicy(pi_dev, policy.c), t_max).value(beta).mean() <= ev.value(beta).mean() + 1e-8


def test_push_beats_pull_theorem_2():
    """Theorem 2: the optimal push policy is at least as good as the optimal pull policy.

    API finds a Nash equilibrium, not necessarily the optimum, so we take the
    better of the two initialisations and compare with the best pull policy found.
    """
    mdp, beta, t_max = control_mdp(4, n_states=5, density=0.5), 0.3, 4
    _, pull_ev = pull.mpi(mdp, beta, t_max, restarts=30)
    best_push = max(
        push.api(mdp, beta, t_max, c0)[1].value(beta).mean()
        for c0 in (push.never_transmit(t_max, 5), push.always_transmit(t_max, 5))
    )
    assert best_push >= pull_ev.value(beta).mean() - 0.05  # allow for API's local equilibria


def test_estimation_pull_policy_guesses_most_likely_state():
    mdp = random_estimation_mdp(6, 0.3, 0.9, np.random.default_rng(0))
    policy, ev = pull.mpi(mdp, 0.2, 4, restarts=5)
    omega = pull._beliefs(mdp, policy.pi_d, 4)
    for l in range(6):
        for k in range(policy.tau[l]):
            assert policy.pi_d[k, l] == np.argmax(omega[l, k])
