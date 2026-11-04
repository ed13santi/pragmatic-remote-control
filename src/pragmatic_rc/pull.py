"""Pull-based remote control (single decision-maker, the decoder).

The decoder tracks ``(delta, l)``: ``l`` is the last state it received and
``delta`` the number of steps since. Its policy is a pair

* ``tau[l]`` in ``1..Tmax``: after receiving state ``l`` it waits ``tau[l]``
  steps and then requests a new update (``tau = 1`` means every step);
* ``pi_d[delta, l]``: the control action taken ``delta`` steps after receiving ``l``.

Because the policy is deterministic, the decoder's belief about the current
state is fixed by ``(delta, l)`` and the policy: it is ``e_l`` propagated
through the transition matrices of the actions already taken. Each request costs
``beta``; the first state is revealed for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import MDP

TOL = 1e-12


@dataclass
class PullPolicy:
    tau: np.ndarray  # (S,)   ints in 1..Tmax
    pi_d: np.ndarray  # (Tmax, S) ints in 0..A-1

    def copy(self) -> "PullPolicy":
        return PullPolicy(self.tau.copy(), self.pi_d.copy())


@dataclass
class Evaluation:
    """Discounted reward and discounted number of transmissions, per start state."""

    reward: np.ndarray  # (S,)
    tx: np.ndarray  # (S,)

    def value(self, beta: float) -> np.ndarray:
        return self.reward - beta * self.tx


def random_policy(mdp: MDP, t_max: int, rng: np.random.Generator) -> PullPolicy:
    return PullPolicy(
        tau=rng.integers(1, t_max + 1, size=mdp.n_states),
        pi_d=rng.integers(0, mdp.n_actions, size=(t_max, mdp.n_states)),
    )


def _beliefs(mdp: MDP, pi_d: np.ndarray, t_max: int) -> np.ndarray:
    """``omega[l, k, :]``: belief ``k`` steps after receiving ``l``, shape (S, Tmax, S)."""
    S = mdp.n_states
    omega = np.zeros((S, t_max, S))
    omega[:, 0, :] = np.eye(S)
    for k in range(t_max - 1):
        omega[:, k + 1, :] = np.einsum("lx,lxy->ly", omega[:, k, :], mdp.P[pi_d[k]])
    return omega


def evaluate(mdp: MDP, policy: PullPolicy, t_max: int) -> Evaluation:
    """Exact policy evaluation by solving ``V0 = G + M V0`` over the renewal states."""
    S, g = mdp.n_states, mdp.gamma
    omega = _beliefs(mdp, policy.pi_d, t_max)
    G_r = np.zeros(S)
    G_c = np.zeros(S)
    M = np.zeros((S, S))
    for k in range(t_max):
        active = k < policy.tau
        a = policy.pi_d[k]
        G_r += active * g**k * (omega[:, k, :] * mdp.Rbar[a]).sum(axis=1)
        stop = policy.tau == k + 1
        nxt = np.einsum("lx,lxy->ly", omega[:, k, :], mdp.P[a])
        M += stop[:, None] * g ** (k + 1) * nxt
        G_c += stop * g ** (k + 1)
    A = np.eye(S) - M
    return Evaluation(reward=np.linalg.solve(A, G_r), tx=np.linalg.solve(A, G_c))


def _improve(
    mdp: MDP,
    policy: PullPolicy,
    v0: np.ndarray,
    beta: float,
    t_max: int,
    fixed_tau: int | None,
) -> PullPolicy:
    """One sweep of policy improvement for every last-received state."""
    S, A, g = mdp.n_states, mdp.n_actions, mdp.gamma
    new = policy.copy()
    for l in range(S):
        old_a = policy.pi_d[:, l]
        # vec[k][x]: value of being at step k with true state x and then
        # following the old plan (old actions, request at the old waiting time).
        # Past the old waiting time the plan is hypothetical: it keeps the old
        # actions and requests when forced at step Tmax.
        vec = np.zeros((t_max, S))
        for k in range(t_max - 1, -1, -1):
            a = old_a[k]
            if k == t_max - 1 or k + 1 == policy.tau[l]:
                vec[k] = mdp.Rbar[a] + g * (mdp.P[a] @ v0 - beta)
            else:
                vec[k] = mdp.Rbar[a] + g * (mdp.P[a] @ vec[k + 1])

        belief = np.zeros(S)
        belief[l] = 1.0
        for k in range(t_max):
            options = []
            for a in [old_a[k]] + [x for x in range(A) if x != old_a[k]]:
                base = belief @ mdp.Rbar[a]
                stop_v = base + g * (belief @ (mdp.P[a] @ v0) - beta)
                cont_v = base + g * (belief @ (mdp.P[a] @ vec[k + 1])) if k < t_max - 1 else -np.inf
                if fixed_tau is not None:
                    if k + 1 == fixed_tau:
                        cont_v = -np.inf
                    else:
                        stop_v = -np.inf
                options.append((stop_v, a, True))
                options.append((cont_v, a, False))
            # Prefer the old decision on ties so that the iteration terminates.
            old_stop = policy.tau[l] == k + 1
            options.sort(key=lambda o: 0 if (o[1] == old_a[k] and o[2] == old_stop) else 1)
            best = options[0]
            for opt in options[1:]:
                if opt[0] > best[0] + TOL:
                    best = opt
            _, a, stop = best
            new.pi_d[k, l] = a
            if stop:
                new.tau[l] = k + 1
                break
            belief = belief @ mdp.P[a]
    return new


def _same(a: PullPolicy, b: PullPolicy) -> bool:
    if not np.array_equal(a.tau, b.tau):
        return False
    return all(np.array_equal(a.pi_d[: a.tau[l], l], b.pi_d[: b.tau[l], l]) for l in range(len(a.tau)))


def _mpi_single(
    mdp: MDP, beta: float, t_max: int, policy: PullPolicy, fixed_tau: int | None, max_iter: int
) -> tuple[PullPolicy, Evaluation]:
    for _ in range(max_iter):
        ev = evaluate(mdp, policy, t_max)
        new = _improve(mdp, policy, ev.value(beta), beta, t_max, fixed_tau)
        if _same(policy, new):
            return policy, ev
        policy = new
    raise RuntimeError("MPI did not converge")


def mpi(
    mdp: MDP,
    beta: float,
    t_max: int,
    init: PullPolicy | None = None,
    fixed_tau: int | None = None,
    restarts: int = 1,
    rng: np.random.Generator | None = None,
    max_iter: int = 500,
) -> tuple[PullPolicy, Evaluation]:
    """Modified policy iteration for the pull-based problem.

    Alternates exact evaluation with a sweep that, for every last-received
    state, re-selects the control action at each delay and the waiting time.
    The value never decreases and the iteration terminates, but the fixed point
    is a local optimum: the decoder's plan is open-loop between requests, so
    the problem is not convex in the action sequence. ``restarts > 1`` runs
    from several random initial policies and keeps the best one.
    ``fixed_tau`` restricts the waiting time to a constant (periodic baseline).
    """
    rng = rng or np.random.default_rng(0)
    best = None
    for r in range(restarts):
        policy = init.copy() if (init is not None and r == 0) else random_policy(mdp, t_max, rng)
        if fixed_tau is not None:
            policy.tau[:] = fixed_tau
        result = _mpi_single(mdp, beta, t_max, policy, fixed_tau, max_iter)
        if best is None or result[1].value(beta).mean() > best[1].value(beta).mean() + 1e-12:
            best = result
    return best


def brute_force_value(mdp: MDP, beta: float, t_max: int, tol: float = 1e-12) -> np.ndarray:
    """Optimal value ``V0*`` over all open-loop plans, by value iteration.

    For every last-received state it enumerates every action sequence of length
    ``1..Tmax``. Exponential in ``Tmax``; intended as a ground-truth check on
    small instances.
    """
    from itertools import product

    S, g = mdp.n_states, mdp.gamma
    plans = []  # (G, gamma^len * next-state distribution) per start state
    for l in range(S):
        entries = []
        for length in range(1, t_max + 1):
            for seq in product(range(mdp.n_actions), repeat=length):
                b = np.zeros(S)
                b[l] = 1.0
                gain = 0.0
                for k, a in enumerate(seq):
                    gain += g**k * (b @ mdp.Rbar[a])
                    b = b @ mdp.P[a]
                entries.append((gain - g**length * beta, g**length * b))
        plans.append(entries)
    v = np.zeros(S)
    while True:
        new = np.array([max(gain + w @ v for gain, w in plans[l]) for l in range(S)])
        if np.max(np.abs(new - v)) < tol:
            return new
        v = new
