"""Remote control of a noisy inverted pendulum, as a Gymnasium environment for PPO.

The encoder sees the pendulum. The decoder applies the torque and only learns
the state when the encoder pays ``beta`` to send it. This wraps the coordinator
problem of :mod:`pragmatic_rc.optimal` as a single-agent environment so that a
stock PPO implementation can solve it.

One time step is two decisions, and the observation says which one is due:

* **decoder phase.** The observation is a summary of the decoder's belief. The
  agent's first action component is the torque.
* **encoder phase.** The observation adds the true state. The second action
  component is the send decision (send if it is positive).

In the wrong phase the other component is ignored. Because the policy is only
shown what the acting agent may know (the true state is zeroed in the decoder
phase), execution is decentralised even though a single network is trained.

The decoder's belief is a particle set that the environment maintains. After a
silent step each particle is reweighted by the probability that the encoder would
have stayed quiet there, which needs the current policy: call ``set_policy``
with the PPO model's policy before training.
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

G, M, L, DT, MAX_SPEED, MAX_TORQUE = 10.0, 1.0, 1.0, 0.05, 8.0, 2.0
SINCE_CAP = 10


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def features(p: np.ndarray) -> np.ndarray:
    """(theta, omega) -> (cos, sin, omega / 8), shape (..., 3)."""
    return np.stack([np.cos(p[..., 0]), np.sin(p[..., 0]), p[..., 1] / MAX_SPEED], axis=-1)


def dynamics(p: np.ndarray, u: float, noise: float, rng) -> np.ndarray:
    th, om = p[..., 0], p[..., 1]
    om = om + (3 * G / (2 * L) * np.sin(th) + 3.0 / (M * L**2) * u) * DT
    om = np.clip(om + noise * rng.standard_normal(om.shape), -MAX_SPEED, MAX_SPEED)
    return np.stack([th + om * DT, om], axis=-1)


class RemoteControlPendulum(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, beta=0.3, horizon=100, noise=0.3, n_particles=64, force_send=None, reward_scale=0.1):
        self.beta, self.horizon, self.noise, self.N = beta, horizon, noise, n_particles
        self.force_send, self.scale = force_send, reward_scale
        self.observation_space = spaces.Box(-np.inf, np.inf, (11,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.policy = None
        self.stochastic_filter = True  # False: use the hard rule, for deterministic evaluation
        self.rng = np.random.default_rng()

    def set_policy(self, policy) -> None:
        self.policy = policy

    # -- belief
    def _summary(self) -> np.ndarray:
        f = features(self.P)
        return np.concatenate([f.mean(0), f.std(0), [min(self.since, SINCE_CAP) / SINCE_CAP]])

    def _obs(self, b, x_feat, phase) -> np.ndarray:
        return np.concatenate([b, x_feat, [phase]]).astype(np.float32)

    def _quiet_weights(self, b2: np.ndarray) -> np.ndarray:
        """Probability, for every particle, that the encoder would have stayed silent there."""
        if self.force_send == 0:
            return np.ones(self.N)
        if self.policy is None:
            return np.full(self.N, 0.5)
        obs = np.concatenate([np.tile(b2, (self.N, 1)), features(self.P), np.ones((self.N, 1))], axis=1)
        with torch.no_grad():
            d = self.policy.get_distribution(torch.as_tensor(obs, dtype=torch.float32)).distribution
            mu, sd = d.mean[:, 1], d.stddev[:, 1]
            p_send = torch.special.ndtr(mu / sd).numpy()
        if not self.stochastic_filter:
            p_send = (p_send > 0.5).astype(float)
        return 1.0 - p_send

    # -- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x = np.array([self.rng.uniform(-np.pi, np.pi), self.rng.uniform(-1, 1)])
        self.P = np.tile(self.x, (self.N, 1))  # the first state is revealed for free
        self.since, self.t, self.phase = 0, 0, 0
        self.received = self.x.copy()
        return self._obs(self._summary(), np.zeros(3), 0), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        if self.phase == 0:
            u = MAX_TORQUE * a[0]
            cost = angle_normalize(self.x[0]) ** 2 + 0.1 * self.x[1] ** 2 + 0.001 * u**2
            self.x = dynamics(self.x, u, self.noise, self.rng)
            self.P = dynamics(self.P, u, self.noise, self.rng)
            self.since += 1
            self.t += 1
            self.b2 = self._summary()
            self.phase = 1
            info = dict(cost=cost, sent=False, x=self.x.copy(), u=u)
            return self._obs(self.b2, features(self.x), 1), -cost * self.scale, False, False, info
        send = bool(a[1] > 0) if self.force_send is None else bool(self.force_send)
        w = self._quiet_weights(self.b2)
        if send:
            self.P = np.tile(self.x, (self.N, 1))
            self.since = 0
            self.received = self.x.copy()
        else:
            if w.sum() < 1e-8:
                w = np.ones(self.N)
            self.P = self.P[self.rng.choice(self.N, self.N, p=w / w.sum())]
        self.phase = 0
        truncated = self.t >= self.horizon
        info = dict(cost=0.0, sent=send, x=self.x.copy(), received=self.received.copy())
        return self._obs(self._summary(), np.zeros(3), 0), -self.beta * self.scale * send, False, truncated, info


# ---- rendering ---------------------------------------------------------------


def draw_pendulum(ax, theta: float, color: str = "#1f3b73") -> None:
    tip = (np.sin(theta), np.cos(theta))
    ax.plot([0, tip[0]], [0, tip[1]], lw=7, color=color, solid_capstyle="round", zorder=2)
    ax.add_patch(plt_circle((0, 0), 0.06, "#333"))
    ax.add_patch(plt_circle(tip, 0.13, "#c0392b"))


def plt_circle(xy, r, color):
    from matplotlib.patches import Circle

    return Circle(xy, r, color=color, zorder=3)


def _background(ax, color: str) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((-1.5, -1.5), 3, 3, color=color, zorder=0))


def render_gif(frames, path: str, fps: int = 20) -> None:
    """``frames`` is a list of dicts with ``x`` (true state), ``received`` (state just
    sent, or None when silent), ``t`` and ``sends`` (cumulative)."""
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = []
    for f in frames:
        fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.9), dpi=80)
        for ax in (left, right):
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.axis("off")
        _background(right, "#f4f4f2")
        draw_pendulum(right, f["x"][0])
        right.set_title("Pendulum", fontsize=11)
        if f["received"] is None:
            _background(left, "black")
            left.set_title("Controller view: no signal", fontsize=11)
        else:
            _background(left, "#f4f4f2")
            draw_pendulum(left, f["received"][0])
            left.set_title("Controller view: update received", fontsize=11, color="#1e8449")
        fig.suptitle(f"t = {f['t']:3d}   transmissions: {f['sends']}", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, images, duration=1000 / fps, loop=0)
