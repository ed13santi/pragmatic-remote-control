"""Finite MDPs and the benchmark families used in the experiments.

Conventions
-----------
``P[a, s, s']`` is the transition probability, ``R[a, s, s']`` the reward
received when the process moves from ``s`` to ``s'`` under action ``a``
(the paper's ``r_{s_t, s_{t+1}}(a_t)``), and ``gamma`` the discount factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class MDP:
    P: np.ndarray
    R: np.ndarray
    gamma: float

    def __post_init__(self) -> None:
        if self.P.ndim != 3 or self.P.shape[1] != self.P.shape[2]:
            raise ValueError("P must have shape (A, S, S)")
        if self.R.shape != self.P.shape:
            raise ValueError("R must have the same shape as P")
        if not np.allclose(self.P.sum(axis=-1), 1.0):
            raise ValueError("rows of P must sum to one")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")

    @property
    def n_actions(self) -> int:
        return self.P.shape[0]

    @property
    def n_states(self) -> int:
        return self.P.shape[1]

    @property
    def Rbar(self) -> np.ndarray:
        """Expected one-step reward, shape (A, S)."""
        return (self.P * self.R).sum(axis=-1)


def _spread_kernel(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangular kernel over ``m`` consecutive states, peaked at the centre."""
    idx = np.arange(m)
    weights = (m + 1) / 2 - np.abs(idx - (m - 1) / 2)
    return idx - (m - 1) // 2, weights / weights.sum()


def _spread_transitions(base: np.ndarray, n_states: int, density: float) -> np.ndarray:
    """Turn deterministic transitions ``base[a, s]`` into a matrix of given density.

    The probability mass of each deterministic transition is spread over
    ``m = round(density * S)`` neighbouring states (circular index distance),
    decreasing linearly with the distance from the original target. This follows
    the construction described in Sec. VI of the paper, but the exact
    coefficients differ from the closed form printed there.
    """
    m = int(np.clip(round(density * n_states), 1, n_states))
    offsets, weights = _spread_kernel(m)
    n_actions = base.shape[0]
    P = np.zeros((n_actions, n_states, n_states))
    for a in range(n_actions):
        for s in range(n_states):
            for off, w in zip(offsets, weights):
                P[a, s, (base[a, s] + off) % n_states] += w
    return P


def random_control_mdp(
    n_states: int,
    n_actions: int,
    density: float,
    gamma: float,
    rng: np.random.Generator,
    hi_reward: float = 1.0,
    lo_reward: float = 0.5,
) -> MDP:
    """Two-peak control task.

    A random deterministic MDP is made stochastic by spreading each transition
    over ``density * n_states`` neighbours. Reaching the target state yields
    ``hi_reward``; reaching a second state or either of its neighbours yields
    ``lo_reward``.
    """
    base = rng.integers(0, n_states, size=(n_actions, n_states))
    P = _spread_transitions(base, n_states, density)
    reward_of_next = np.zeros(n_states)
    s_hi, s_lo = n_states // 4, (3 * n_states) // 4
    for k in (-1, 0, 1):
        reward_of_next[(s_lo + k) % n_states] = lo_reward
    reward_of_next[s_hi] = hi_reward
    R = np.broadcast_to(reward_of_next, P.shape).copy()
    return MDP(P=P, R=R, gamma=gamma)


def random_estimation_mdp(
    n_states: int,
    density: float,
    gamma: float,
    rng: np.random.Generator,
) -> MDP:
    """Remote estimation: the decoder guesses the state, reward 1 if correct.

    Actions are guesses (``A = S``) and do not influence the dynamics.
    """
    base = rng.integers(0, n_states, size=(1, n_states))
    P0 = _spread_transitions(base, n_states, density)[0]
    P = np.broadcast_to(P0, (n_states, n_states, n_states)).copy()
    R = np.zeros_like(P)
    for a in range(n_states):
        R[a, a, :] = 1.0
    return MDP(P=P, R=R, gamma=gamma)
