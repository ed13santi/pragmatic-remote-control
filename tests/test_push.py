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


def exhaustive_decoder_value(mdp, c, beta, t_max):
    """Independent reference: best decoder value against encoder ``c`` by trying every action sequence."""
    from itertools import product

    S, g = mdp.n_states, mdp.gamma
    plans = []
    for l in range(S):
        entries = []
        for seq in product(range(mdp.n_actions), repeat=t_max):
            u = np.eye(S)[l]
            gain, mass = 0.0, np.zeros(S)
            for k, a in enumerate(seq):
                gain += g**k * (u @ mdp.Rbar[a])
                moved = u @ mdp.P[a]
                sent = moved * c[k + 1, l]
                gain -= beta * g ** (k + 1) * sent.sum()
                mass += g ** (k + 1) * sent
                u = moved * (1 - c[k + 1, l])
            entries.append((gain, mass))
        plans.append(entries)
    v = np.zeros(S)
    while True:
        new = np.array([max(gain + mass @ v for gain, mass in plans[l]) for l in range(S)])
        if np.max(np.abs(new - v)) < 1e-12:
            return new
        v = new


@pytest.mark.parametrize("seed", range(12))
def test_decoder_best_response_is_optimal_in_control(seed):
    """In control the decoder's belief depends on its own action history, so greedy
    policy iteration over (delta, last state) can fail; the exact search must not."""
    mdp, beta, t_max = control_mdp(seed, n_states=4, density=0.4), 0.2, 4
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 2, size=(t_max + 1, 4, 4))
    c[0], c[t_max] = 0, 1
    pi_d = push.decoder_best_response(mdp, rng.integers(0, 2, size=(t_max, 4)), c, beta, t_max)
    achieved = push.evaluate(mdp, push.PushPolicy(pi_d, c), t_max).value(beta)
    np.testing.assert_allclose(achieved, exhaustive_decoder_value(mdp, c, beta, t_max), atol=1e-8)


def test_greedy_decoder_step_is_exact_only_without_action_dependence():
    """Estimation: greedy equals exact. Control: it is allowed to be worse."""
    est = random_estimation_mdp(4, 0.4, 0.9, np.random.default_rng(0))
    rng, t_max, beta = np.random.default_rng(1), 3, 0.2
    c = rng.integers(0, 2, size=(t_max + 1, 4, 4))
    c[0], c[t_max] = 0, 1
    start = rng.integers(0, 4, size=(t_max, 4))
    greedy = push._decoder_best_response_local(est, start, c, beta, t_max)
    value = push.evaluate(est, push.PushPolicy(greedy, c), t_max).value(beta)
    np.testing.assert_allclose(value, exhaustive_decoder_value(est, c, beta, t_max), atol=1e-8)


def test_greedy_api_can_miss_the_decoder_best_response_in_control():
    """Regression for the control case: with greedy policy iteration over (delta, last state)
    the decoder step is not a best response, and API ends in a worse fixed point."""
    mdp, beta, t_max = control_mdp(12, n_states=4, density=0.4), 0.2, 4
    start = push.always_transmit(t_max, 4)
    greedy_policy, greedy = push.api(mdp, beta, t_max, start, exact=False)
    exact_policy, exact = push.api(mdp, beta, t_max, start)
    best_vs_greedy_encoder = exhaustive_decoder_value(mdp, greedy_policy.c, beta, t_max)
    assert np.max(best_vs_greedy_encoder - greedy.value(beta)) > 0.05  # greedy is not a best response
    assert exact.value(beta).mean() > greedy.value(beta).mean() + 0.01


@pytest.mark.parametrize("init", ["never", "always"])
def test_api_reaches_a_nash_equilibrium(init):
    mdp, beta, t_max = control_mdp(3), 0.2, 4
    c0 = (push.never_transmit if init == "never" else push.always_transmit)(t_max, mdp.n_states)
    policy, ev = push.api(mdp, beta, t_max, c0)
    # Neither player can improve by deviating unilaterally.
    c_dev = push.encoder_best_response(mdp, policy.pi_d, policy.c, beta, t_max)
    assert push.evaluate(mdp, push.PushPolicy(policy.pi_d, c_dev), t_max).value(beta).mean() <= ev.value(beta).mean() + 1e-8
    best = exhaustive_decoder_value(mdp, policy.c, beta, t_max)
    assert np.all(best <= ev.value(beta) + 1e-8)


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
