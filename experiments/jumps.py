"""Learned encoder versus threshold encoders on the cart-pole variants with jumps.

usage: python experiments/jumps.py TASK KIND [VALUE] [timesteps]

TASK    inverted_pendulum_kicks or inverted_pendulum_switching
KIND    always            send every step
        never             never send (the controller runs open loop on its prediction)
        joint             learn when to send (belief summary: mean and spread)
        joint_dist        as joint, with the distance of x from the belief mean given as an input
        joint_rich        as joint, with 10% and 90% quantiles in the belief summary
        threshold_learned the policy picks the threshold at every step from the belief
                          (a state-dependent threshold); only the threshold and controller are learned
        threshold VALUE   send when the largest feature error against the belief mean
                          exceeds VALUE; only the controller is learned

Score is steps survived (of 200) minus beta per transmission (higher is better).
"""
import json, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for the GIF
import torch
from pragmatic_rc import ppo
from pragmatic_rc.mujoco_env import RemoteControlMujoco

torch.set_num_threads(1)
task, kind = sys.argv[1], sys.argv[2]
value = float(sys.argv[3]) if kind == "threshold" else 0.0
steps = int(sys.argv[-1]) if len(sys.argv) > (4 if kind == "threshold" else 3) else 500_000
kw = dict(task=task)
if kind == "always":
    kw["force_send"] = 1
elif kind == "never":
    kw["force_send"] = 0
elif kind == "joint_dist":
    kw["dist_input"] = True
elif kind == "joint_rich":
    kw["rich"] = True
elif kind == "threshold":
    kw["threshold"] = value
elif kind == "threshold_learned":
    kw["threshold"] = "learned"

ent = float(os.environ.get("ENT_COEF", 0.0))  # optional entropy bonus
seed = int(os.environ.get("SEED", 0))
model = ppo.train(kw, total_timesteps=steps, seed=seed, verbose=1, env_cls=RemoteControlMujoco, ent_coef=ent)
cost, se, sends = ppo.evaluate(model, kw, episodes=50, env_cls=RemoteControlMujoco)
print("RESULT", task, kind, value, -cost, se, sends, flush=True)
tag = f"jumps_{task}_{kind}_{value}" + (f"_steps{steps}_ent{ent}" if (steps != 500_000 or ent) else "") + (f"_seed{seed}" if seed else "")
json.dump(dict(task=task, kind=kind, value=value, score=-cost, se=se, sends=sends, steps=steps), open(f"figures/{tag}.json", "w"))
model.save(f"figures/{tag}.zip")
if kind != "always" and os.environ.get("GIF"):
    from pragmatic_rc.mujoco_env import render_gif
    _, _, frames = ppo.run_episode(model, kw, seed=7, log=True, env_cls=RemoteControlMujoco)
    render_gif(frames, f"figures/{tag}.gif", task=task)
