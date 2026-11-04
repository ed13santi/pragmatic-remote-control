"""Re-render the GIF of a saved joint cart-pole policy: python experiments/render_cartpole.py [beta] [seed]"""
import sys
from stable_baselines3 import PPO
from pragmatic_rc import ppo
from pragmatic_rc.mujoco_env import RemoteControlInvertedPendulum as Env, render_gif

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
model = PPO.load(f"figures/cartpole_joint_{beta}.zip", device="cpu")
cost, sends, frames = ppo.run_episode(model, dict(beta=beta), seed=seed, log=True, env_cls=Env)
render_gif(frames, f"figures/cartpole_{beta}.gif")
print("score", -cost, "sends", sends, "frames", len(frames))
