"""POMCP for the coordinator POMDP, for state spaces too large to enumerate.

:mod:`pragmatic_rc.optimal` solves the joint problem exactly but enumerates
every set of states in which the encoder might transmit (``2^S`` of them) and
every belief vector. Neither scales. This module plans with Monte Carlo tree
search instead (POMCP, Silver and Veness 2010) and needs only a simulator.

Two changes make it fit this problem.

**A small family of encoder rules.** The coordinator cannot pick an arbitrary
subset of a huge state space. Instead it chooses among a finite set of rules
(:class:`RuleClass`) that map the true state and the decoder's current belief
to a transmit decision, for example "transmit if the state is further than
``theta`` from where the decoder believes it is". The optimum over this family
is a lower bound on the true optimum; :class:`AllSubsets` recovers the full
problem on small instances.

**Renewal nodes.** A transmission reveals the state, after which the belief is a
point mass and the future depends only on that state. All histories that end in
the same transmitted state therefore share one subtree. This turns the search
tree into a graph, and lets value estimates for ``V(e_s)`` accumulate across
the whole run.

Each time step has two phases, as in :mod:`pragmatic_rc.optimal`: the encoder's
rule is chosen first (``W`` nodes), then the decoder acts after seeing whether
a message arrived (``V`` nodes).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Hashable

import numpy as np

from .envs import Simulator

MAX_PARTICLES = 400


# ---- encoder rule families -----------------------------------------------------


class Belief:
    """A bounded particle set with cached summaries used by the encoder rules."""

    def __init__(self, sim: Simulator, rng: np.random.Generator, particles=None):
        self.sim, self.rng = sim, rng
        self.particles: list = list(particles) if particles is not None else []
        self.n_seen = len(self.particles)
        self._stamp = -1
        self._mean = None
        self._hist = None

    def add(self, s) -> None:
        self.n_seen += 1
        if len(self.particles) < MAX_PARTICLES:
            self.particles.append(s)
        else:
            j = int(self.rng.integers(self.n_seen))
            if j < MAX_PARTICLES:
                self.particles[j] = s

    def _refresh(self) -> None:
        if self._stamp != self.n_seen:
            self._stamp = self.n_seen
            self._mean = None
            self._hist = None

    def mean_features(self) -> np.ndarray:
        self._refresh()
        if self._mean is None:
            self._mean = np.mean([self.sim.features(s) for s in self.particles], axis=0)
        return self._mean

    def mass(self, s) -> float:
        self._refresh()
        if self._hist is None:
            self._hist = Counter(self.particles)
        return self._hist[s] / len(self.particles)


class RuleClass:
    """A finite family of encoder rules. ``transmit(k, s, belief)`` applies rule ``k``."""

    n_rules: int

    def transmit(self, k: int, s: Hashable, belief: Belief) -> bool:
        raise NotImplementedError

    def describe(self, k: int) -> str:
        return f"rule {k}"


class DistanceFromMean(RuleClass):
    """Transmit if the state is further than ``theta`` from the belief mean (max-norm of the features).

    ``theta < 0`` always transmits and ``theta = inf`` never does.
    """

    def __init__(self, thetas):
        self.thetas = list(thetas)
        self.n_rules = len(self.thetas)

    def transmit(self, k, s, belief):
        theta = self.thetas[k]
        if theta < 0:
            return True
        if math.isinf(theta):
            return False
        return float(np.max(np.abs(belief.sim.features(s) - belief.mean_features()))) > theta

    def describe(self, k):
        return f"distance > {self.thetas[k]}"


class BeliefMassThreshold(RuleClass):
    """Transmit if the decoder gives the true state less than ``theta`` of its belief.

    Natural for estimation: the states worth reporting are the ones the decoder
    considers unlikely. ``theta = 0`` never transmits; ``theta > 1`` always does.
    """

    def __init__(self, thetas):
        self.thetas = list(thetas)
        self.n_rules = len(self.thetas)

    def transmit(self, k, s, belief):
        return belief.mass(s) < self.thetas[k]

    def describe(self, k):
        return f"belief mass < {self.thetas[k]}"


class AllSubsets(RuleClass):
    """Every subset of ``n_states`` states, as in the exact solver. Only for tiny problems."""

    def __init__(self, n_states: int):
        self.table = np.array(list(product([0, 1], repeat=n_states)), dtype=bool)
        self.n_rules = len(self.table)

    def transmit(self, k, s, belief):
        return bool(self.table[k, s])

    def describe(self, k):
        return "send in " + str(np.flatnonzero(self.table[k]).tolist())


# ---- search tree ----------------------------------------------------------------


class Edge:
    __slots__ = ("n", "q", "silent")

    def __init__(self):
        self.n = 0
        self.q = 0.0
        self.silent = None  # child V node after a silent step (W edges only)


class Node:
    __slots__ = ("kind", "n", "edges", "belief", "next")

    def __init__(self, kind: str, belief: Belief):
        self.kind, self.n, self.edges, self.belief = kind, 0, None, belief
        self.next: dict[int, "Node"] = {}  # V nodes: control action -> W node


@dataclass
class POMCP:
    sim: Simulator
    rules: RuleClass
    beta: float
    n_sims: int = 500
    c: float = 1.0
    max_depth: int | None = None
    rollout_rule: Callable | None = None  # (s, belief) -> bool; default never transmit
    seed: int = 0
    rng: np.random.Generator = field(init=False)
    renewal: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        if self.max_depth is None:
            self.max_depth = int(math.ceil(math.log(0.02) / math.log(self.sim.gamma)))

    # -- nodes
    def renewal_node(self, s) -> Node:
        """The shared ``V`` node for 'the decoder has just been told the state is ``s``'."""
        node = self.renewal.get(s)
        if node is None:
            node = self.renewal[s] = Node("V", Belief(self.sim, self.rng, [s]))
        return node

    def _ucb(self, node: Node) -> int:
        untried = [i for i, e in enumerate(node.edges) if e.n == 0]
        if untried:
            return int(self.rng.choice(untried))
        log_n = math.log(node.n + 1)
        scores = [e.q + self.c * math.sqrt(log_n / e.n) for e in node.edges]
        return int(np.argmax(scores))

    def _init_edges(self, node: Node) -> None:
        n = self.sim.n_actions if node.kind == "V" else self.rules.n_rules
        node.edges = [Edge() for _ in range(n)]

    # -- simulation
    def _rollout(self, s, depth: int) -> float:
        g, total, disc = self.sim.gamma, 0.0, 1.0
        for _ in range(depth, self.max_depth):
            s, r = self.sim.step(s, int(self.rng.integers(self.sim.n_actions)), self.rng)
            total += disc * r
            disc *= g
        return total

    def _sim_v(self, node: Node, s, depth: int) -> float:
        if depth > self.max_depth:
            return 0.0
        if node.edges is None:
            self._init_edges(node)
            return self._rollout(s, depth)
        a = self._ucb(node)
        edge = node.edges[a]
        s2, r = self.sim.step(s, a, self.rng)
        child = node.next.get(a)
        if child is None:
            child = node.next[a] = Node("W", Belief(self.sim, self.rng))
        child.belief.add(s2)
        ret = r + self.sim.gamma * self._sim_w(child, s2, depth + 1)
        node.n += 1
        edge.n += 1
        edge.q += (ret - edge.q) / edge.n
        return ret

    def _sim_w(self, node: Node, s, depth: int) -> float:
        if depth > self.max_depth:
            return 0.0
        if node.edges is None:
            self._init_edges(node)
            return self._rollout(s, depth)
        k = self._ucb(node)
        edge = node.edges[k]
        if self.rules.transmit(k, s, node.belief):
            cost, child = -self.beta, self.renewal_node(s)
        else:
            cost = 0.0
            if edge.silent is None:
                edge.silent = Node("V", Belief(self.sim, self.rng))
            child = edge.silent
            child.belief.add(s)
        ret = cost + self._sim_v(child, s, depth)
        node.n += 1
        edge.n += 1
        edge.q += (ret - edge.q) / edge.n
        return ret

    # -- planning
    def plan(self, node: Node) -> int:
        """Run simulations from ``node`` and return the index of the best action there."""
        if node.edges is None:
            self._init_edges(node)
        sim_fn = self._sim_v if node.kind == "V" else self._sim_w
        for _ in range(self.n_sims):
            s = node.belief.particles[int(self.rng.integers(len(node.belief.particles)))]
            sim_fn(node, s, 0)
        return int(np.argmax([e.q if e.n else -np.inf for e in node.edges]))


# ---- running the planner against a real system ---------------------------------


@dataclass
class EpisodeResult:
    reward: float
    transmissions: float  # discounted count
    sends: int  # raw count
    steps: int


def run_episode(agent: POMCP, horizon: int, rng: np.random.Generator) -> EpisodeResult:
    """Run one episode against the simulator, planning online.

    The initial state is revealed for free. Encoder and decoder share the
    planner's tree, which stands in for their common knowledge.
    """
    sim, g = agent.sim, agent.sim.gamma
    s = sim.sample_initial(rng)
    node = agent.renewal_node(s)
    reward = tx = 0.0
    sends = 0
    for t in range(horizon):
        # decoder phase
        a = agent.plan(node)
        s2, r = sim.step(s, a, rng)
        reward += g**t * r
        w_node = node.next.get(a)
        if w_node is None:
            w_node = node.next[a] = Node("W", Belief(sim, agent.rng, node.belief.particles[:1]))
        w_node.belief.add(s2)
        s = s2
        # encoder phase
        k = agent.plan(w_node)
        if agent.rules.transmit(k, s, w_node.belief):
            tx += g ** (t + 1)
            sends += 1
            node = agent.renewal_node(s)
        else:
            edge = w_node.edges[k]
            if edge.silent is None:
                edge.silent = Node("V", Belief(sim, agent.rng))
            node = edge.silent
            node.belief.add(s)
    return EpisodeResult(reward, tx, sends, horizon)
