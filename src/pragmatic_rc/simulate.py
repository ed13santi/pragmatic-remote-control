"""Monte Carlo simulation of the full encoder-decoder system.

These simulators never compute a belief. They run the true Markov chain with
tabular policies, so they are an independent check on the analytic evaluation
in :mod:`pragmatic_rc.pull` and :mod:`pragmatic_rc.push`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import MDP
from .pull import PullPolicy
from .push import PushPolicy


@dataclass
class SimResult:
    reward: float
    tx: float
    reward_se: float
    tx_se: float


def _sample_next(mdp: MDP, a: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cum = np.cumsum(mdp.P[a, s], axis=-1)
    nxt = (rng.random(len(s))[:, None] > cum).sum(axis=1)
    return np.minimum(nxt, mdp.n_states - 1)


def _horizon(gamma: float) -> int:
    return int(np.ceil(np.log(1e-7) / np.log(gamma)))


def _summarise(reward: np.ndarray, tx: np.ndarray) -> SimResult:
    n = len(reward)
    return SimResult(reward.mean(), tx.mean(), reward.std(ddof=1) / np.sqrt(n), tx.std(ddof=1) / np.sqrt(n))


def simulate_pull(mdp: MDP, policy: PullPolicy, n: int, rng: np.random.Generator) -> SimResult:
    g = mdp.gamma
    s = rng.integers(0, mdp.n_states, size=n)
    last, delta = s.copy(), np.zeros(n, dtype=int)
    reward, tx = np.zeros(n), np.zeros(n)
    for t in range(_horizon(g)):
        a = policy.pi_d[delta, last]
        s2 = _sample_next(mdp, a, s, rng)
        reward += g**t * mdp.R[a, s, s2]
        delta += 1
        request = delta == policy.tau[last]
        tx += g ** (t + 1) * request
        last = np.where(request, s2, last)
        delta = np.where(request, 0, delta)
        s = s2
    return _summarise(reward, tx)


def simulate_push(mdp: MDP, policy: PushPolicy, n: int, rng: np.random.Generator) -> SimResult:
    g = mdp.gamma
    s = rng.integers(0, mdp.n_states, size=n)
    last, delta = s.copy(), np.zeros(n, dtype=int)
    reward, tx = np.zeros(n), np.zeros(n)
    for t in range(_horizon(g)):
        a = policy.pi_d[delta, last]
        s2 = _sample_next(mdp, a, s, rng)
        reward += g**t * mdp.R[a, s, s2]
        delta += 1
        send = policy.c[delta, last, s2] == 1
        tx += g ** (t + 1) * send
        last = np.where(send, s2, last)
        delta = np.where(send, 0, delta)
        s = s2
    return _summarise(reward, tx)
