"""Learned encoder versus threshold encoders on the cart-pole variants with jumps.

usage: python experiments/jumps.py TASK KIND [VALUE] [timesteps]

TASK    inverted_pendulum_kicks or inverted_pendulum_switching
KIND    always            send every step
        joint             learn when to send (belief summary: mean and spread)
        joint_rich        as joint, with 10% and 90% quantiles in the belief summary
        threshold_learned the policy picks the threshold at every step from the belief
                          (a state-dependent threshold); only the threshold and controller are learned
        threshold VALUE   send when the largest feature error against the belief mean
                          exceeds VALUE; only the controller is learned

Score is steps survived (of 200) minus beta per transmission (higher is better).
"""
import json, os, sys
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
elif kind == "joint_rich":
    kw["rich"] = True
elif kind == "threshold":
    kw["threshold"] = value
elif kind == "threshold_learned":
    kw["threshold"] = "learned"

ent = float(os.environ.get("ENT_COEF", 0.0))  # optional entropy bonus
model = ppo.train(kw, total_timesteps=steps, verbose=1, env_cls=RemoteControlMujoco, ent_coef=ent)
cost, se, sends = ppo.evaluate(model, kw, episodes=50, env_cls=RemoteControlMujoco)
print("RESULT", task, kind, value, -cost, se, sends, flush=True)
tag = f"jumps_{task}_{kind}_{value}" + (f"_steps{steps}_ent{ent}" if (steps != 500_000 or ent) else "")
json.dump(dict(task=task, kind=kind, value=value, score=-cost, se=se, sends=sends, steps=steps), open(f"figures/{tag}.json", "w"))
