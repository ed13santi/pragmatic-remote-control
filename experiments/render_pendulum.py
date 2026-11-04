"""Re-render the GIF of a saved joint pendulum policy: python experiments/render_pendulum.py [beta] [seed]"""
import sys
from stable_baselines3 import PPO
from pragmatic_rc import ppo
from pragmatic_rc.pendulum import render_gif

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
model = PPO.load(f"figures/pendulum_joint_{beta}.zip", device="cpu")
_, sends, frames = ppo.run_episode(model, dict(beta=beta), seed=seed, log=True)
render_gif(frames, f"figures/pendulum_{beta}.gif")
print("sends", sends)
