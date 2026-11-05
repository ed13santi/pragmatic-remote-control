"""Remote control of a MuJoCo task with PPO.

usage: python experiments/mujoco_task.py TASK {joint,never,always} [beta] [timesteps]

TASK is inverted_pendulum, inverted_double_pendulum, reacher or hopper. Score is
the task's return minus beta per transmission (higher is better). `joint` also
writes a GIF of one episode.
"""
import json, sys
import torch
from pragmatic_rc import ppo
from pragmatic_rc.mujoco_env import RemoteControlMujoco, TASKS, render_gif

torch.set_num_threads(1)
task, variant = sys.argv[1], sys.argv[2]
beta = float(sys.argv[3]) if len(sys.argv) > 3 else TASKS[task].beta
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1_000_000
kw = dict(task=task, beta=beta, force_send={"joint": None, "never": 0, "always": 1}[variant])

model = ppo.train(kw, total_timesteps=steps, verbose=1, env_cls=RemoteControlMujoco)
cost, se, sends = ppo.evaluate(model, kw, episodes=50, env_cls=RemoteControlMujoco)
print("RESULT", task, variant, beta, -cost, se, sends, flush=True)
tag = f"{task}_{variant}_{beta}"
json.dump(dict(task=task, variant=variant, beta=beta, score=-cost, se=se, sends=sends, steps=steps), open(f"figures/{tag}.json", "w"))
model.save(f"figures/{tag}.zip")
if variant == "joint":
    _, _, frames = ppo.run_episode(model, kw, seed=7, log=True, env_cls=RemoteControlMujoco)
    render_gif(frames, f"figures/{task}_{beta}.gif", task=task)
