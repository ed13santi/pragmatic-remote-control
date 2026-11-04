"""Push-based remote control (encoder and decoder are two decision-makers).

The decoder state is ``(delta, l)`` as in the pull-based case. The encoder sees
the true state ``x`` as well and, ``delta`` steps after its last transmission of
``l``, decides whether to send the current state:

* ``c[delta, l, x]`` in {0, 1} for ``delta`` in ``1..Tmax`` (``c[Tmax] = 1``: a
  transmission is forced once ``Tmax`` steps have elapsed);
* ``pi_d[delta, l]`` is the decoder's control action for ``delta`` in ``0..Tmax-1``.

Silence carries information. If the decoder knows ``c``, observing no
transmission rules out every state in which the encoder would have sent. The
decoder's belief therefore follows the filtering recursion of Eq. (5) in the
paper, which is what makes this problem harder than the pull-based one.

Alternating policy iteration (API) improves the decoder against a fixed
encoder, then the encoder against a fixed decoder, until neither moves. The
fixed point is a Nash equilibrium of the common-payoff game.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import MDP
from .pull import Evaluation

TOL = 1e-9


@dataclass
class PushPolicy:
    pi_d: np.ndarray  # (Tmax, S)         decoder action
    c: np.ndarray  # (Tmax + 1, S, S)  encoder transmit decision, index [delta, l, x]

    def copy(self) -> "PushPolicy":
        return PushPolicy(self.pi_d.copy(), self.c.copy())


def never_transmit(t_max: int, n_states: int) -> np.ndarray:
    """Encoder that stays silent until the forced transmission at ``Tmax``."""
    c = np.zeros((t_max + 1, n_states, n_states), dtype=int)
    c[t_max] = 1
    return c


def always_transmit(t_max: int, n_states: int) -> np.ndarray:
    return np.ones((t_max + 1, n_states, n_states), dtype=int)


def _decoder_masses(mdp: MDP, pi_d: np.ndarray, c: np.ndarray, t_max: int):
    """Unnormalised belief ``u[l, k, :]`` and transmitted mass ``tx[l, k, :]``.

    ``u[l, k, x]`` is the probability that the state is ``x`` ``k`` steps after
    ``l`` was received *and* nothing was sent in between. ``tx[l, k, y]`` is the
    probability that the encoder sends at step ``k + 1`` while in state ``y``.
    """
    S = mdp.n_states
    u = np.zeros((S, t_max, S))
    tx = np.zeros((S, t_max, S))
    u[:, 0, :] = np.eye(S)
    for k in range(t_max):
        moved = np.einsum("lx,lxy->ly", u[:, k, :], mdp.P[pi_d[k]])
        tx[:, k, :] = moved * c[k + 1]
        if k + 1 < t_max:
            u[:, k + 1, :] = moved * (1 - c[k + 1])
    return u, tx


def evaluate(mdp: MDP, policy: PushPolicy, t_max: int) -> Evaluation:
    S, g = mdp.n_states, mdp.gamma
    u, tx = _decoder_masses(mdp, policy.pi_d, policy.c, t_max)
    G_r = np.zeros(S)
    G_c = np.zeros(S)
    M = np.zeros((S, S))
    for k in range(t_max):
        G_r += g**k * (u[:, k, :] * mdp.Rbar[policy.pi_d[k]]).sum(axis=1)
        M += g ** (k + 1) * tx[:, k, :]
        G_c += g ** (k + 1) * tx[:, k, :].sum(axis=1)
    A = np.eye(S) - M
    return Evaluation(reward=np.linalg.solve(A, G_r), tx=np.linalg.solve(A, G_c))


def _improve_decoder(mdp: MDP, policy: PushPolicy, v0: np.ndarray, beta: float, t_max: int) -> np.ndarray:
    S, A, g = mdp.n_states, mdp.n_actions, mdp.gamma
    c = policy.c
    new = policy.pi_d.copy()
    for l in range(S):
        old_a = policy.pi_d[:, l]
        # Value of what follows step k+1, as a function of the true state y there.
        after = np.zeros((t_max, S))
        vec = np.zeros((t_max, S))
        for k in range(t_max - 1, -1, -1):
            cont = vec[k + 1] if k + 1 < t_max else np.zeros(S)
            after[k] = c[k + 1, l] * (v0 - beta) + (1 - c[k + 1, l]) * cont
            a = old_a[k]
            vec[k] = mdp.Rbar[a] + g * (mdp.P[a] @ after[k])
        u = np.zeros(S)
        u[l] = 1.0
        for k in range(t_max):
            best_a, best_v = old_a[k], u @ mdp.Rbar[old_a[k]] + g * (u @ (mdp.P[old_a[k]] @ after[k]))
            for a in range(A):
                v = u @ mdp.Rbar[a] + g * (u @ (mdp.P[a] @ after[k]))
                if v > best_v + TOL:
                    best_a, best_v = a, v
            new[k, l] = best_a
            u = (u @ mdp.P[best_a]) * (1 - c[k + 1, l])
    return new


def decoder_best_response(
    mdp: MDP, pi_d: np.ndarray, c: np.ndarray, beta: float, t_max: int, max_iter: int = 500
) -> np.ndarray:
    pi_d = pi_d.copy()
    for _ in range(max_iter):
        policy = PushPolicy(pi_d, c)
        v0 = evaluate(mdp, policy, t_max).value(beta)
        new = _improve_decoder(mdp, policy, v0, beta, t_max)
        if np.array_equal(new, pi_d):
            return pi_d
        pi_d = new
    raise RuntimeError("decoder best response did not converge")


def encoder_best_response(
    mdp: MDP, pi_d: np.ndarray, c_old: np.ndarray, beta: float, t_max: int, tol: float = 1e-11
) -> np.ndarray:
    """Optimal encoder against a fixed decoder: value iteration on ``(x, delta, l)``."""
    S, g = mdp.n_states, mdp.gamma
    P_a = mdp.P[pi_d]  # (Tmax, l, x, y)
    R_a = mdp.Rbar[pi_d]  # (Tmax, l, x)
    idx = np.arange(S)
    ve = np.zeros((t_max + 1, S, S))  # [delta, l, x]

    def q_values(ve: np.ndarray) -> np.ndarray:
        return R_a + g * np.einsum("dlxy,dly->dlx", P_a, ve[1 : t_max + 1])

    for _ in range(100_000):
        q = q_values(ve)
        send = q[0][idx, idx] - beta  # decoder state (0, x) right after sending x
        new = np.empty_like(ve)
        new[1:t_max] = np.maximum(send[None, None, :], q[1:t_max])
        new[t_max] = send[None, :]
        new[0] = 0.0
        if np.max(np.abs(new - ve)) < tol:
            ve = new
            break
        ve = new
    q = q_values(ve)
    send = q[0][idx, idx] - beta
    c = c_old.copy()
    gap = send[None, None, :] - q[1:t_max]
    c[1:t_max] = np.where(gap > TOL, 1, np.where(gap < -TOL, 0, c_old[1:t_max]))
    c[t_max] = 1
    return c


def api(
    mdp: MDP,
    beta: float,
    t_max: int,
    c_init: np.ndarray,
    rng: np.random.Generator | None = None,
    max_iter: int = 500,
) -> tuple[PushPolicy, Evaluation]:
    """Alternating policy iteration from an initial encoder policy."""
    rng = rng or np.random.default_rng(0)
    pi_d = rng.integers(0, mdp.n_actions, size=(t_max, mdp.n_states))
    c = c_init.copy()
    for _ in range(max_iter):
        new_pi_d = decoder_best_response(mdp, pi_d, c, beta, t_max)
        new_c = encoder_best_response(mdp, new_pi_d, c, beta, t_max)
        if np.array_equal(new_pi_d, pi_d) and np.array_equal(new_c, c):
            policy = PushPolicy(pi_d, c)
            return policy, evaluate(mdp, policy, t_max)
        pi_d, c = new_pi_d, new_c
    raise RuntimeError("API did not converge")
