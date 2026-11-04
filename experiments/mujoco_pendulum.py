"""Remote control of a MuJoCo cart-pole with PPO.

usage: python experiments/mujoco_pendulum.py {joint,never,always} [beta] [timesteps]

Score is steps survived (out of 200) minus beta per transmission; higher is better.
"""
import json, sys
import torch
from pragmatic_rc import ppo
from pragmatic_rc.mujoco_env import RemoteControlInvertedPendulum as Env, render_gif

torch.set_num_threads(1)
variant = sys.argv[1]
beta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
kw = dict(beta=beta, force_send={"joint": None, "never": 0, "always": 1}[variant])

model = ppo.train(kw, total_timesteps=steps, verbose=1, env_cls=Env)
cost, se, sends = ppo.evaluate(model, kw, episodes=100, env_cls=Env)
print("RESULT", variant, beta, -cost, se, sends, flush=True)
tag = f"cartpole_{variant}_{beta}"
json.dump(dict(variant=variant, beta=beta, score=-cost, se=se, sends=sends), open(f"figures/{tag}.json", "w"))
model.save(f"figures/{tag}.zip")
if variant == "joint":
    _, _, frames = ppo.run_episode(model, kw, seed=7, log=True, env_cls=Env)
    render_gif(frames, f"figures/cartpole_{beta}.gif")
