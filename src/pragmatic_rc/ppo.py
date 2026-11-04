"""PPO (Stable-Baselines3) training and evaluation for the remote-control environments."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from .pendulum import RemoteControlPendulum


def train(env_kwargs: dict, total_timesteps: int = 600_000, n_envs: int = 8, seed: int = 0, verbose: int = 1):
    """Train one PPO agent that plays both encoder and decoder."""
    vec = VecMonitor(DummyVecEnv([lambda: RemoteControlPendulum(**env_kwargs) for _ in range(n_envs)]))
    model = PPO(
        "MlpPolicy", vec, device="cpu", seed=seed, verbose=verbose,
        n_steps=256, batch_size=256, n_epochs=10, learning_rate=5e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.0,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-0.5),
    )
    for e in vec.venv.envs:
        e.unwrapped.set_policy(model.policy)  # the filter needs the policy being trained
    model.learn(total_timesteps=total_timesteps)
    return model


def run_episode(model, env_kwargs: dict, seed: int, log: bool = False):
    """One deterministic episode. Returns (cost, sends[, frames]); cost is in pendulum units."""
    env = RemoteControlPendulum(**env_kwargs)
    env.stochastic_filter = False
    env.set_policy(model.policy)
    obs, _ = env.reset(seed=seed)
    cost = sends = 0.0
    frames = [dict(x=env.x.copy(), received=env.received.copy(), t=0, sends=0)] if log else None
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, done, info = env.step(action)
        if env.phase == 1:
            cost += info["cost"]
        else:
            cost += env.beta * info["sent"]
            sends += info["sent"]
            if log:
                frames.append(dict(x=info["x"], received=info["received"] if info["sent"] else None, t=env.t, sends=int(sends)))
    return (cost, sends, frames) if log else (cost, sends)


def evaluate(model, env_kwargs: dict, episodes: int = 50, seed: int = 1000):
    """Mean total cost (pendulum cost plus beta per transmission), standard error, sends per episode."""
    res = np.array([run_episode(model, env_kwargs, seed + i) for i in range(episodes)])
    return res[:, 0].mean(), res[:, 0].std() / np.sqrt(episodes), res[:, 1].mean()
