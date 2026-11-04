"""Remote control of a noisy pendulum with PPO.

usage: python experiments/pendulum.py {joint,never,always} [beta] [timesteps]

Trains one variant, evaluates it and writes figures/pendulum_<variant>.json.
`joint` learns when to send and how to control; `never` and `always` fix the
encoder and train the controller alone. `joint` also writes a GIF of an episode.
"""
import json, sys
import torch
from pragmatic_rc import ppo
from pragmatic_rc.pendulum import render_gif

torch.set_num_threads(1)
variant = sys.argv[1]
beta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
kw = dict(beta=beta, force_send={"joint": None, "never": 0, "always": 1}[variant])

model = ppo.train(kw, total_timesteps=steps, verbose=1)
mean, se, sends = ppo.evaluate(model, kw, episodes=100)
print("RESULT", variant, beta, mean, se, sends, flush=True)
json.dump(dict(variant=variant, beta=beta, cost=mean, se=se, sends=sends), open(f"figures/pendulum_{variant}_{beta}.json", "w"))
model.save(f"figures/pendulum_{variant}_{beta}.zip")
if variant == "joint":
    _, _, frames = ppo.run_episode(model, kw, seed=7, log=True)
    render_gif(frames, f"figures/pendulum_{beta}.gif")
