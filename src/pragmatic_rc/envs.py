"""Generative models for sampling-based planning.

POMCP never looks at transition matrices. It only needs to *sample* the
process, so any simulator that implements :class:`Simulator` can be plugged in,
including ones whose state space is far too large to enumerate.

A simulator provides:

``sample_initial(rng)``  a starting state;
``step(s, a, rng)``      the next state and the reward for taking ``a`` in ``s``;
``features(s)``          a vector describing ``s``, used by distance-based
                         encoder rules (see :mod:`pragmatic_rc.pomcp`).

States must be hashable, because a transmitted state becomes part of the
planner's history.
"""

from __future__ import annotations

from typing import Hashable, Protocol

import numpy as np

from .mdp import MDP


class Simulator(Protocol):
    n_actions: int
    gamma: float

    def sample_initial(self, rng: np.random.Generator) -> Hashable: ...

    def step(self, s: Hashable, a: int, rng: np.random.Generator) -> tuple[Hashable, float]: ...

    def features(self, s: Hashable) -> np.ndarray: ...


class TabularSimulator:
    """Wraps an :class:`MDP`, so that sampling-based and exact solvers can be compared."""

    def __init__(self, mdp: MDP, onehot: bool = False) -> None:
        self.onehot = onehot
        self.feature_scale = 1.0 if onehot else float(mdp.n_states)
        self.mdp = mdp
        self.n_actions = mdp.n_actions
        self.n_states = mdp.n_states
        self.gamma = mdp.gamma
        self._cum = np.cumsum(mdp.P, axis=-1)

    def sample_initial(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.n_states))

    def step(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, float]:
        s2 = min(int(np.searchsorted(self._cum[a, s], rng.random(), side="right")), self.n_states - 1)
        return s2, float(self.mdp.R[a, s, s2])

    def features(self, s: int) -> np.ndarray:
        if self.onehot:
            return np.eye(self.n_states)[s]
        return np.array([float(s)])

    # vectorised versions used by the learned policies
    def sample_initial_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(self.n_states, size=n)

    def step_batch(self, s: np.ndarray, a: np.ndarray, rng: np.random.Generator):
        c = self._cum[a, s]
        s2 = np.minimum((c <= rng.random(len(s))[:, None]).sum(axis=1), self.n_states - 1)
        return s2, self.mdp.R[a, s, s2].astype(float)

    def features_batch(self, s: np.ndarray) -> np.ndarray:
        if self.onehot:
            return np.eye(self.n_states)[s]
        return s[:, None].astype(float)


class GridWorld:
    """Navigation on an ``n`` by ``n`` grid with slippery moves.

    Actions are stay, up, down, left, right. With probability ``slip`` the
    agent moves in a uniformly random direction instead. The reward is 1 for
    every step that ends on the goal cell. States are cell indices, so the
    state space grows as ``n ** 2``.
    """

    MOVES = np.array([[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]])

    def __init__(self, n: int = 8, slip: float = 0.2, gamma: float = 0.9, goal: tuple[int, int] | None = None):
        self.n, self.slip, self.gamma = n, slip, gamma
        self.n_actions = 5
        self.n_states = n * n
        self.goal = goal if goal is not None else (n - 1, n - 1)
        self.feature_scale = float(n - 1)

    def _cell(self, s: int) -> tuple[int, int]:
        return s // self.n, s % self.n

    def sample_initial(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.n_states))

    def step(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, float]:
        if rng.random() < self.slip:
            a = int(rng.integers(1, 5))
        x, y = self._cell(s)
        dx, dy = self.MOVES[a]
        x, y = int(np.clip(x + dx, 0, self.n - 1)), int(np.clip(y + dy, 0, self.n - 1))
        return x * self.n + y, float((x, y) == self.goal)

    def features(self, s: int) -> np.ndarray:
        return np.array(self._cell(s), dtype=float)

    def sample_initial_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(self.n_states, size=n)

    def step_batch(self, s: np.ndarray, a: np.ndarray, rng: np.random.Generator):
        a = np.where(rng.random(len(s)) < self.slip, rng.integers(1, 5, size=len(s)), a)
        x = np.clip(s // self.n + self.MOVES[a, 0], 0, self.n - 1)
        y = np.clip(s % self.n + self.MOVES[a, 1], 0, self.n - 1)
        return x * self.n + y, ((x == self.goal[0]) & (y == self.goal[1])).astype(float)

    def features_batch(self, s: np.ndarray) -> np.ndarray:
        return np.stack([s // self.n, s % self.n], axis=1).astype(float)

    def to_mdp(self) -> MDP:
        """Exact transition and reward tensors, for baselines that need them."""
        S, A = self.n_states, self.n_actions
        P = np.zeros((A, S, S))
        R = np.zeros((A, S, S))
        for s in range(S):
            x, y = self._cell(s)
            for a in range(A):
                outcomes = [(a, 1 - self.slip)] + [(b, self.slip / 4) for b in range(1, 5)]
                for b, p in outcomes:
                    dx, dy = self.MOVES[b]
                    nx, ny = int(np.clip(x + dx, 0, self.n - 1)), int(np.clip(y + dy, 0, self.n - 1))
                    s2 = nx * self.n + ny
                    P[a, s, s2] += p
                    R[a, s, s2] = float((nx, ny) == self.goal)
        return MDP(P=P, R=R, gamma=self.gamma)
