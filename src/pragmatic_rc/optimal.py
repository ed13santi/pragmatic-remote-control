"""Optimal joint policy via heuristic search value iteration (HSVI) on beliefs.

Idea
----
The encoder and decoder share all the information the decoder has. A fictitious
coordinator that sees only this common information can therefore prescribe, at
every step, a *rule* for the encoder (which states to transmit) and an action
for the decoder. The belief ``b`` over the current state, built from everything
the decoder has observed *and every control action it has taken*, is a
sufficient statistic for this coordinator. This turns the two-agent problem
into a single-agent POMDP and removes the need to track the time since the last
update or the action history explicitly: both are absorbed into ``b``.

Each time step has two phases, which keeps the encoder's rule from depending on
information the decoder does not have yet:

``W`` phase, belief ``b`` over the current state, before the encoder acts.
    The coordinator picks the set ``T`` of states in which the encoder will
    transmit, paying ``beta`` per transmission. The decoder observes the state
    if it is in ``T`` and otherwise learns that it is not in ``T``.

``V`` phase, belief ``b'`` after that observation.
    The decoder picks a control action ``a``, collects the expected reward, and
    the belief moves to ``b' P^a``, which starts the next ``W`` phase.

Both value functions are convex and piecewise linear in the belief. HSVI keeps
a lower bound (a set of alpha vectors) and an upper bound (a sawtooth
interpolation of sampled points) for each, and refines both along trajectories
that the upper bound deems promising, until their gap at the starting beliefs
is below ``eps``. The result is an ``eps``-optimal value together with a policy
that can be executed by tracking the belief.

The initial state is revealed for free, as in :mod:`pragmatic_rc.pull` and
:mod:`pragmatic_rc.push`, so the quantity reported is
``mean_s V(e_s)`` and values are directly comparable.

The cost is exponential in the number of states (the coordinator enumerates the
``2^S`` subsets of states to transmit), so this is a reference solver for small
problems rather than a method for large ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .mdp import MDP


class LowerBound:
    """Maximum of alpha vectors."""

    def __init__(self, n_states: int, floor: float) -> None:
        self.alphas = np.full((1, n_states), floor)

    def eval(self, B: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(B) @ self.alphas.T).max(axis=1)

    def best_alpha(self, b: np.ndarray) -> np.ndarray:
        return self.alphas[np.argmax(self.alphas @ b)]

    def add(self, alpha: np.ndarray) -> None:
        if not np.any(np.all(np.abs(self.alphas - alpha) < 1e-12, axis=1)):
            self.alphas = np.vstack([self.alphas, alpha])


class UpperBound:
    """Sawtooth interpolation between corner values and sampled belief points."""

    def __init__(self, n_states: int, ceiling: float) -> None:
        self.corners = np.full(n_states, ceiling)
        self.points = np.empty((0, n_states))
        self.values = np.empty(0)

    def eval(self, B: np.ndarray) -> np.ndarray:
        B = np.atleast_2d(B)
        base = B @ self.corners
        if len(self.values) == 0:
            return base
        slack = self.values - self.points @ self.corners  # <= 0 for useful points
        support = self.points > 0
        safe = np.where(support, self.points, 1.0)
        ratio = np.where(support[None], B[:, None, :] / safe[None], np.inf).min(axis=2)
        return np.minimum(base, (base[:, None] + ratio * slack[None]).min(axis=1))

    def add(self, b: np.ndarray, value: float) -> None:
        if value < self.eval(b)[0] - 1e-12:
            self.points = np.vstack([self.points, b])
            self.values = np.append(self.values, value)
        if np.count_nonzero(b) == 1:  # corner: tighten directly
            s = int(np.argmax(b))
            self.corners[s] = min(self.corners[s], value)


@dataclass
class Solution:
    lb_w: LowerBound
    lb_v: LowerBound
    ub_w: UpperBound
    ub_v: UpperBound
    lower: np.ndarray  # V_lb(e_s), shape (S,)
    upper: np.ndarray  # V_ub(e_s), shape (S,)
    mdp: MDP
    beta: float
    eps: float
    iterations: int = 0
    subsets: np.ndarray = field(repr=False, default=None)

    @property
    def value_bounds(self) -> tuple[float, float]:
        """Bounds on the optimal expected return, uniform initial state revealed for free."""
        return float(self.lower.mean()), float(self.upper.mean())


class _Solver:
    def __init__(self, mdp: MDP, beta: float, eps: float) -> None:
        self.mdp, self.beta, self.eps = mdp, beta, eps
        S, g = mdp.n_states, mdp.gamma
        self.S = S
        self.eye = np.eye(S)
        self.Rbar = mdp.Rbar
        # Prescriptions: C[k, s] = 1 if the encoder transmits in state s.
        self.C = np.array(list(product([0, 1], repeat=S)), dtype=float)
        r_lo, r_hi = float(self.Rbar.min()), float(self.Rbar.max())
        self.lb_v = LowerBound(S, r_lo / (1 - g))
        self.lb_w = LowerBound(S, r_lo / (1 - g))
        self.ub_v = UpperBound(S, r_hi / (1 - g))
        self.ub_w = UpperBound(S, r_hi / (1 - g))
        self.iterations = 0

    # ---- one-step lookaheads -------------------------------------------------

    def q_w(self, b: np.ndarray, v_bound) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Value of every prescription at ``W``-belief ``b``, using bound ``v_bound`` for ``V``.

        Returns the values, the probability of staying silent and the belief
        after silence, for every prescription.
        """
        corner = v_bound.eval(self.eye)
        tx = (self.C * b) @ (corner - self.beta)
        silent_mass = (1 - self.C) @ b
        ok = silent_mass > 1e-15
        b_silent = np.zeros_like(self.C)
        b_silent[ok] = ((1 - self.C) * b)[ok] / silent_mass[ok, None]
        silent_value = np.zeros(len(self.C))
        if ok.any():
            silent_value[ok] = v_bound.eval(b_silent[ok])
        return tx + silent_mass * silent_value, silent_mass, b_silent

    def q_v(self, b: np.ndarray, w_bound) -> np.ndarray:
        """Value of every control action at ``V``-belief ``b``, using bound ``w_bound`` for ``W``."""
        nxt = np.einsum("s,ast->at", b, self.mdp.P)
        return self.Rbar @ b + self.mdp.gamma * w_bound.eval(nxt)

    # ---- backups -------------------------------------------------------------

    def backup_w(self, b: np.ndarray) -> None:
        q_lb, _, b_silent = self.q_w(b, self.lb_v)
        k = int(np.argmax(q_lb))
        c = self.C[k]
        corner = self.lb_v.eval(self.eye)
        alpha = np.where(c > 0, corner - self.beta, self.lb_v.best_alpha(b_silent[k]))
        self.lb_w.add(alpha)
        q_ub, _, _ = self.q_w(b, self.ub_v)
        self.ub_w.add(b, float(q_ub.max()))

    def backup_v(self, b: np.ndarray) -> None:
        g, P = self.mdp.gamma, self.mdp.P
        q_lb = self.q_v(b, self.lb_w)
        a = int(np.argmax(q_lb))
        alpha_next = self.lb_w.best_alpha(b @ P[a])
        self.lb_v.add(self.Rbar[a] + g * P[a] @ alpha_next)
        q_ub = self.q_v(b, self.ub_w)
        self.ub_v.add(b, float(q_ub.max()))

    # ---- exploration ---------------------------------------------------------

    def width_v(self, b: np.ndarray) -> float:
        return float(self.ub_v.eval(b)[0] - self.lb_v.eval(b)[0])

    def width_w(self, b: np.ndarray) -> float:
        return float(self.ub_w.eval(b)[0] - self.lb_w.eval(b)[0])

    def explore_w(self, b: np.ndarray, t: int) -> None:
        g, tol = self.mdp.gamma, self.eps / self.mdp.gamma**t
        if self.width_w(b) <= tol:
            return
        q_ub, silent_mass, b_silent = self.q_w(b, self.ub_v)
        k = int(np.argmax(q_ub))
        children = [(float(b[s]), self.eye[s]) for s in range(self.S) if self.C[k, s] > 0 and b[s] > 0]
        if silent_mass[k] > 1e-15:
            children.append((float(silent_mass[k]), b_silent[k]))
        excess = [p * (self.width_v(child) - tol) for p, child in children]
        j = int(np.argmax(excess))
        if excess[j] > 0:
            self.explore_v(children[j][1], t)
        self.backup_w(b)

    def explore_v(self, b: np.ndarray, t: int) -> None:
        g, tol = self.mdp.gamma, self.eps / self.mdp.gamma**t
        if self.width_v(b) <= tol:
            return
        a = int(np.argmax(self.q_v(b, self.ub_w)))
        nxt = b @ self.mdp.P[a]
        if self.width_w(nxt) > self.eps / g ** (t + 1):
            self.explore_w(nxt, t + 1)
        self.backup_v(b)

    def run(self, max_trials: int) -> Solution:
        S = self.S
        for trial in range(max_trials):
            widths = [self.width_v(self.eye[s]) for s in range(S)]
            if max(widths) <= self.eps:
                break
            self.explore_v(self.eye[int(np.argmax(widths))], 0)
            self.iterations += 1
        lower = self.lb_v.eval(self.eye)
        upper = self.ub_v.eval(self.eye)
        return Solution(
            self.lb_w, self.lb_v, self.ub_w, self.ub_v, lower, upper, self.mdp, self.beta, self.eps,
            self.iterations, self.C,
        )


def solve(mdp: MDP, beta: float, eps: float = 1e-2, max_trials: int = 20_000) -> Solution:
    """Solve the joint problem to within ``eps`` at every revealed initial state."""
    return _Solver(mdp, beta, eps).run(max_trials)


# ---- executing the policy ----------------------------------------------------


def simulate(sol: Solution, n: int, rng: np.random.Generator, horizon: int | None = None):
    """Run the lower-bound policy from random initial states by tracking the belief.

    Returns the mean discounted reward and the mean discounted number of
    transmissions, each with a standard error.
    """
    mdp, beta = sol.mdp, sol.beta
    S, g = mdp.n_states, mdp.gamma
    horizon = horizon or int(np.ceil(np.log(1e-5) / np.log(g)))
    solver = _Solver(mdp, beta, sol.eps)
    solver.lb_v, solver.lb_w = sol.lb_v, sol.lb_w
    reward, tx = np.zeros(n), np.zeros(n)
    for i in range(n):
        s = int(rng.integers(S))
        b = np.eye(S)[s]
        for t in range(horizon):
            a = int(np.argmax(solver.q_v(b, sol.lb_w)))
            s2 = int(rng.choice(S, p=mdp.P[a, s]))
            reward[i] += g**t * mdp.R[a, s, s2]
            b = b @ mdp.P[a]
            q, _, b_silent = solver.q_w(b, sol.lb_v)
            k = int(np.argmax(q))
            if solver.C[k, s2] > 0:
                tx[i] += g ** (t + 1)
                b = np.eye(S)[s2]
            else:
                b = b_silent[k]
            s = s2
    se = lambda x: x.std(ddof=1) / np.sqrt(n)
    return reward.mean(), tx.mean(), se(reward), se(tx)
