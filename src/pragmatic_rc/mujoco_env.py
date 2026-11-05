"""Remote control of MuJoCo tasks, as Gymnasium environments for PPO.

Same construction as :mod:`pragmatic_rc.pendulum`: one agent plays encoder and
decoder in alternating phases, and the environment keeps the decoder's belief as
a particle set that is reweighted by the probability that the encoder stayed
silent. The physics is MuJoCo; the true state and all particles advance together
in one call to ``mujoco.rollout``.

Randomness comes from noise on the actuators, which the real system and every
particle experience. The first state is revealed for free. The score of an
episode is the task's return minus ``beta`` per transmission.

Tasks (Gymnasium's v5 models and rewards): ``inverted_pendulum`` (cart-pole),
``inverted_double_pendulum``, ``reacher`` and ``hopper``. Add a task by writing a
:class:`Task`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import mujoco
import numpy as np
import torch
from gymnasium import spaces
from mujoco import rollout

SINCE_CAP = 10


def asset(name: str) -> str:
    return os.path.join(os.path.dirname(gym.__file__), "envs", "mujoco", "assets", name + ".xml")


@dataclass
class Task:
    xml: str
    frame_skip: int
    ctrl_scale: float  # action 1.0 maps to this control value
    noise: float  # actuator noise, in control units
    horizon: int
    beta: float
    reward_scale: float
    n_feat: int
    features: Callable  # (..., nstate) -> (..., n_feat)
    reset: Callable  # (model, rng) -> state
    reward: Callable  # (env, prev_state, new_state, u) -> (reward, terminated)
    camera: Callable  # (state) -> (lookat, distance, azimuth, elevation)
    kick_prob: float = 0.0  # chance per step of a sudden jump in the first velocity
    kick_size: float = 0.0
    switch_prob: float = 0.0  # chance per step that a hidden actuator sign flips (adds a state variable)


# ---- cart-pole -----------------------------------------------------------------

_IP_SCALE = np.array([1.0, 0.2, 3.0, 3.0])


def _ip_reward(env, prev, new, u):
    fell = bool(abs(new[1]) > 0.2)
    return (0.0 if fell else 1.0), fell


# ---- double pendulum -----------------------------------------------------------

_IDP_SCALE = np.array([1.0, 0.3, 0.3, 3.0, 3.0, 3.0])


def _idp_reward(env, prev, new, u):
    env.kin.qpos[:], env.kin.qvel[:] = new[:3], new[3:]
    mujoco.mj_kinematics(env.model, env.kin)
    x, _, y = env.kin.site_xpos[0]
    penalty = 0.01 * x**2 + (y - 2) ** 2 + 1e-3 * new[4] ** 2 + 5e-3 * new[5] ** 2
    terminated = bool(y <= 1)
    return (0.0 if terminated else 10.0 - penalty), terminated


# ---- reacher -------------------------------------------------------------------


def _reacher_tip(s):
    q0, q1 = s[..., 0], s[..., 1]
    return np.stack([0.1 * np.cos(q0) + 0.11 * np.cos(q0 + q1), 0.1 * np.sin(q0) + 0.11 * np.sin(q0 + q1)], axis=-1)


def _reacher_features(s):
    tip, target = _reacher_tip(s), s[..., 2:4]
    return np.concatenate(
        [np.cos(s[..., :2]), np.sin(s[..., :2]), target / 0.2, (tip - target) / 0.2, s[..., 4:6] / 5.0], axis=-1
    )


def _reacher_reset(model, rng):
    qpos = rng.uniform(-0.1, 0.1, model.nq) + model.qpos0
    while True:
        goal = rng.uniform(-0.2, 0.2, 2)
        if np.linalg.norm(goal) < 0.2:
            break
    qpos[2:] = goal
    qvel = 0.005 * rng.standard_normal(model.nv)
    qvel[2:] = 0.0
    return np.concatenate([qpos, qvel])


def _reacher_reward(env, prev, new, u):
    dist = np.linalg.norm(_reacher_tip(new) - new[2:4])
    return -dist - float(np.sum(u**2)), False


# ---- hopper --------------------------------------------------------------------

_HOPPER_SCALE = np.array([1.0, 0.3, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])


def _hopper_features(s):
    return np.clip(s[..., 1:] / _HOPPER_SCALE, -10, 10)  # drop the horizontal position


def _hopper_reward(env, prev, new, u):
    healthy = bool(new[1] > 0.7 and abs(new[2]) < 0.2 and np.all(np.abs(new[2:]) < 100))
    vel = (new[0] - prev[0]) / env.dt
    return vel + (1.0 if healthy else 0.0) - 1e-3 * float(np.sum(u**2)), not healthy


def _hopper_reset(model, rng):
    return np.concatenate([model.qpos0, np.zeros(model.nv)]) + rng.uniform(-0.005, 0.005, model.nq + model.nv)


TASKS = {
    "inverted_pendulum": Task(
        "inverted_pendulum", 2, 3.0, 0.5, 200, 0.5, 0.1, 4, lambda s: s / _IP_SCALE,
        lambda m, r: r.uniform(-0.01, 0.01, 4), _ip_reward, lambda s: ([0, 0, 0.3], 2.6, 90, 0),
    ),
    # Jump variants of the cart-pole: the belief is no longer a single smooth blob.
    "inverted_pendulum_kicks": Task(
        "inverted_pendulum", 2, 3.0, 0.2, 200, 0.5, 0.1, 4, lambda s: s / _IP_SCALE,
        lambda m, r: r.uniform(-0.01, 0.01, 4), _ip_reward, lambda s: ([0, 0, 0.3], 2.6, 90, 0),
        kick_prob=0.05, kick_size=0.6,
    ),
    "inverted_pendulum_switching": Task(
        "inverted_pendulum", 2, 3.0, 0.3, 200, 0.5, 0.1, 5,
        lambda s: np.concatenate([s[..., :4] / _IP_SCALE, s[..., 4:5]], axis=-1),
        lambda m, r: np.concatenate([r.uniform(-0.01, 0.01, 4), [1.0]]), _ip_reward,
        lambda s: ([0, 0, 0.3], 2.6, 90, 0), switch_prob=0.03,
    ),
    "inverted_double_pendulum": Task(
        "inverted_double_pendulum", 5, 1.0, 0.15, 200, 3.0, 0.1, 6, lambda s: s / _IDP_SCALE,
        lambda m, r: np.concatenate([r.uniform(-0.1, 0.1, 3), 0.1 * r.standard_normal(3)]), _idp_reward,
        lambda s: ([0, 0, 0.8], 4.5, 90, 0),
    ),
    "reacher": Task(
        "reacher", 2, 1.0, 0.3, 50, 0.02, 1.0, 10, _reacher_features, _reacher_reset, _reacher_reward,
        lambda s: ([0, 0, 0], 0.9, 90, -90),
    ),
    "hopper": Task(
        "hopper", 4, 1.0, 0.3, 200, 1.0, 0.1, 11, _hopper_features, _hopper_reset, _hopper_reward,
        lambda s: ([float(s[0]), 0, 0.8], 3.5, 90, 0),
    ),
}


class RemoteControlMujoco(gym.Env):
    """Action: the control for each actuator, then a send logit (send if positive)."""

    metadata = {"render_modes": []}

    def __init__(self, task="inverted_pendulum", beta=None, horizon=None, noise=None, n_particles=64,
                 force_send=None, reward_scale=None, threshold=None, rich=False):
        self.task = TASKS[task] if isinstance(task, str) else task
        t = self.task
        self.beta = t.beta if beta is None else beta
        self.horizon = t.horizon if horizon is None else horizon
        self.noise = t.noise if noise is None else noise
        self.scale = t.reward_scale if reward_scale is None else reward_scale
        self.N, self.force_send = n_particles, force_send
        self.threshold, self.rich = threshold, rich  # threshold: send if the feature error exceeds it
        self.model = mujoco.MjModel.from_xml_path(asset(t.xml))
        self.data, self.kin = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.nu = self.model.nu
        self.n_phys = self.model.nq + self.model.nv
        self.dt = self.model.opt.timestep * t.frame_skip
        self.lo, self.hi = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        self.n_summary = (4 if rich else 2) * t.n_feat + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.n_summary + t.n_feat + 1,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu + 1,), np.float32)
        self.policy = None
        self.stochastic_filter = True
        self.rng = np.random.default_rng()

    def set_policy(self, policy) -> None:
        self.policy = policy

    # -- physics
    def _advance(self, P: np.ndarray, u: np.ndarray) -> np.ndarray:
        K, t = len(P), self.task
        init = np.concatenate([np.zeros((K, 1)), P[:, : self.n_phys]], axis=1)
        if t.switch_prob > 0:  # a hidden sign flips the effect of the control
            mode = P[:, self.n_phys]
            drive = u[None, None, :] * mode[:, None, None]
        else:
            drive = u
        ctrl = drive + self.noise * self.rng.standard_normal((K, t.frame_skip, self.nu))
        out, _ = rollout.rollout(self.model, self.data, init, np.clip(ctrl, self.lo, self.hi))
        nxt = out[:, -1, 1:].copy()
        if t.kick_prob > 0:  # rare, sudden jumps in the cart velocity
            kick = self.rng.random(K) < t.kick_prob
            nxt[:, self.model.nq] += kick * self.rng.choice([-1.0, 1.0], K) * t.kick_size
        if t.switch_prob > 0:
            flip = self.rng.random(K) < t.switch_prob
            nxt = np.concatenate([nxt, np.where(flip, -mode, mode)[:, None]], axis=1)
        return nxt

    # -- belief
    def _summary(self) -> np.ndarray:
        f = self.task.features(self.P)
        parts = [f.mean(0), f.std(0)]
        if self.rich:  # quantiles keep some of the shape of a multimodal belief
            parts += [np.quantile(f, 0.1, axis=0), np.quantile(f, 0.9, axis=0)]
        return np.concatenate(parts + [[min(self.since, SINCE_CAP) / SINCE_CAP]])

    def _distance(self, states: np.ndarray) -> np.ndarray:
        """Largest feature error between states and the mean of the belief particles."""
        f = self.task.features(self.P)
        return np.max(np.abs(self.task.features(states) - f.mean(0)), axis=-1)

    def _obs(self, b, x_feat, phase) -> np.ndarray:
        return np.concatenate([b, x_feat, [phase]]).astype(np.float32)

    def _quiet_weights(self, b2: np.ndarray) -> np.ndarray:
        if self.force_send == 0:
            return np.ones(self.N)
        if self.threshold is not None:  # exact: the rule is known, so silence rules out the far particles
            return (self._distance(self.P) <= self.threshold).astype(float)
        if self.policy is None:
            return np.full(self.N, 0.5)
        obs = np.concatenate([np.tile(b2, (self.N, 1)), self.task.features(self.P), np.ones((self.N, 1))], axis=1)
        with torch.no_grad():
            d = self.policy.get_distribution(torch.as_tensor(obs, dtype=torch.float32)).distribution
            p_send = torch.special.ndtr(d.mean[:, -1] / d.stddev[:, -1]).numpy()
        if not self.stochastic_filter:
            p_send = (p_send > 0.5).astype(float)
        return 1.0 - p_send

    # -- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x = self.task.reset(self.model, self.rng)
        self.P = np.tile(self.x, (self.N, 1))  # the first state is revealed for free
        self.since, self.t, self.phase = 0, 0, 0
        self.received = self.x.copy()
        return self._obs(self._summary(), np.zeros(self.task.n_feat), 0), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        if self.phase == 0:
            u = np.clip(self.task.ctrl_scale * a[: self.nu], self.lo, self.hi)
            prev = self.x
            both = self._advance(np.vstack([self.x[None], self.P]), u)
            self.x, self.P = both[0], both[1:]
            self.since += 1
            self.t += 1
            reward, terminated = self.task.reward(self, prev, self.x, u)
            self.b2 = self._summary()
            self.phase = 1
            info = dict(sent=False, x=self.x.copy(), u=u)
            return self._obs(self.b2, self.task.features(self.x), 1), reward * self.scale, terminated, False, info
        if self.threshold is not None:
            send = bool(self._distance(self.x[None])[0] > self.threshold)
        else:
            send = bool(a[-1] > 0) if self.force_send is None else bool(self.force_send)
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
        info = dict(sent=send, x=self.x.copy(), received=self.received.copy())
        return self._obs(self._summary(), np.zeros(self.task.n_feat), 0), -self.beta * self.scale * send, False, truncated, info


class RemoteControlInvertedPendulum(RemoteControlMujoco):
    """The cart-pole task (kept as its own name for the saved cart-pole models)."""

    def __init__(self, **kw):
        super().__init__(task="inverted_pendulum", **kw)


# ---- rendering ---------------------------------------------------------------


def render_gif(frames, path: str, task: str = "inverted_pendulum", fps: int = 25, size=(300, 400)) -> None:
    """Side-by-side GIF. Left: what the controller sees (the state it just received,
    black on steps with no update). Right: the real system."""
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = TASKS[task]
    model = mujoco.MjModel.from_xml_path(asset(spec.xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, *size)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def picture(state):
        data.qpos[:], data.qvel[:] = state[: model.nq], state[model.nq :]
        mujoco.mj_forward(model, data)
        lookat, cam.distance, cam.azimuth, cam.elevation = spec.camera(state)
        cam.lookat[:] = lookat
        renderer.update_scene(data, camera=cam)
        img = renderer.render().copy()
        img[img.max(axis=2) < 12] = (242, 242, 238)  # light background, so a black panel means no signal
        return img

    images = []
    for f in frames:
        fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.3), dpi=80)
        right.imshow(picture(f["x"])); right.set_title("True system", fontsize=11)
        if f["received"] is None:
            left.imshow(np.zeros((size[0], size[1], 3), dtype=np.uint8))
            left.set_title("Controller view: no signal", fontsize=11)
        else:
            left.imshow(picture(f["received"]))
            left.set_title("Controller view: update received", fontsize=11, color="#1e8449")
        for ax in (left, right):
            ax.axis("off")
        fig.suptitle(f"t = {f['t']:3d}   transmissions: {f['sends']}", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    renderer.close()
    imageio.mimsave(path, images, duration=1000 / fps, loop=0)
