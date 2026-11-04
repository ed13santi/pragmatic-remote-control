"""Remote control of a MuJoCo cart-pole (Gymnasium's InvertedPendulum model).

Same construction as :mod:`pragmatic_rc.pendulum`: one agent plays encoder and
decoder in alternating phases, and the environment keeps the decoder's belief as
a particle set that is reweighted by the probability that the encoder stayed
silent. The physics is MuJoCo; all particles are advanced in one call to
``mujoco.rollout``.

The task is to keep the pole upright. Every step the pole is still up earns 1 and
the episode ends when it falls (|angle| > 0.2). The pole is unstable, so the
controller must keep up with the state, but a decoder that models the dynamics
can balance blind for a few steps. Randomness comes from noise on the actuator,
which both the real system and the particles experience.

The score of an episode is steps survived minus ``beta`` per transmission.
"""

from __future__ import annotations

import os

import gymnasium as gym
import mujoco
import numpy as np
import torch
from gymnasium import spaces
from mujoco import rollout

FRAME_SKIP = 2
CTRL_LIMIT = 3.0
ANGLE_LIMIT = 0.2
SCALE = np.array([1.0, 0.2, 3.0, 3.0])  # rough size of (cart x, pole angle, cart v, pole omega)
SINCE_CAP = 10
_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS


def model_path() -> str:
    return os.path.join(os.path.dirname(gym.__file__), "envs", "mujoco", "assets", "inverted_pendulum.xml")


def features(p: np.ndarray) -> np.ndarray:
    return p / SCALE


class RemoteControlInvertedPendulum(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, beta=0.5, horizon=200, noise=0.5, n_particles=64, force_send=None, reward_scale=0.1):
        self.beta, self.horizon, self.noise, self.N = beta, horizon, noise, n_particles
        self.force_send, self.scale = force_send, reward_scale
        self.model = mujoco.MjModel.from_xml_path(model_path())
        self.data = mujoco.MjData(self.model)
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.policy = None
        self.stochastic_filter = True
        self.rng = np.random.default_rng()

    def set_policy(self, policy) -> None:
        self.policy = policy

    # -- physics
    def _advance(self, P: np.ndarray, u: float) -> np.ndarray:
        """Advance a batch of states ``(K, 4)`` by one control step under actuator noise."""
        K = len(P)
        init = np.concatenate([np.zeros((K, 1)), P], axis=1)
        ctrl = np.clip(u + self.noise * self.rng.standard_normal((K, FRAME_SKIP, 1)), -CTRL_LIMIT, CTRL_LIMIT)
        out, _ = rollout.rollout(self.model, self.data, init, ctrl)
        return out[:, -1, 1:5]

    # -- belief
    def _summary(self) -> np.ndarray:
        f = features(self.P)
        return np.concatenate([f.mean(0), f.std(0), [min(self.since, SINCE_CAP) / SINCE_CAP]])

    def _obs(self, b, x_feat, phase) -> np.ndarray:
        return np.concatenate([b, x_feat, [phase]]).astype(np.float32)

    def _quiet_weights(self, b2: np.ndarray) -> np.ndarray:
        if self.force_send == 0:
            return np.ones(self.N)
        if self.policy is None:
            return np.full(self.N, 0.5)
        obs = np.concatenate([np.tile(b2, (self.N, 1)), features(self.P), np.ones((self.N, 1))], axis=1)
        with torch.no_grad():
            d = self.policy.get_distribution(torch.as_tensor(obs, dtype=torch.float32)).distribution
            p_send = torch.special.ndtr(d.mean[:, 1] / d.stddev[:, 1]).numpy()
        if not self.stochastic_filter:
            p_send = (p_send > 0.5).astype(float)
        return 1.0 - p_send

    # -- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x = self.rng.uniform(-0.01, 0.01, size=4)
        self.P = np.tile(self.x, (self.N, 1))  # the first state is revealed for free
        self.since, self.t, self.phase = 0, 0, 0
        self.received = self.x.copy()
        return self._obs(self._summary(), np.zeros(4), 0), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        if self.phase == 0:
            u = CTRL_LIMIT * a[0]
            both = self._advance(np.vstack([self.x[None], self.P]), u)
            self.x, self.P = both[0], both[1:]
            self.since += 1
            self.t += 1
            fell = bool(abs(self.x[1]) > ANGLE_LIMIT)
            self.b2 = self._summary()
            self.phase = 1
            info = dict(sent=False, x=self.x.copy(), u=u)
            return self._obs(self.b2, features(self.x), 1), 0.0 if fell else self.scale, fell, False, info
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
        info = dict(sent=send, x=self.x.copy(), received=self.received.copy())
        return self._obs(self._summary(), np.zeros(4), 0), -self.beta * self.scale * send, False, truncated, info


# ---- rendering ---------------------------------------------------------------


def render_gif(frames, path: str, fps: int = 25, size=(300, 400)) -> None:
    """Side-by-side GIF. Left: what the controller sees (the state it last received,
    shown only on the step it arrives, black otherwise). Right: the real system."""
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = mujoco.MjModel.from_xml_path(model_path())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, *size)
    cam = mujoco.MjvCamera()
    cam.type, cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, [0, 0, 0.3], 2.6, 90, 0

    def picture(state):
        data.qpos[:], data.qvel[:] = state[:2], state[2:]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render().copy()
        img[img.max(axis=2) < 12] = (242, 242, 238)  # light background, so a black panel means no signal
        return img

    images = []
    for f in frames:
        fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.3), dpi=80)
        right.imshow(picture(f["x"])); right.set_title("Cart-pole", fontsize=11)
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
