"""Learned encoder/decoder on a GridWorld too large for the exact solver.

Compares, for several transmission costs: the learned joint policy, the best
decoder that never hears from the encoder, and full-information control paid
for at every step (computed exactly by dynamic programming).
"""
import json, sys
import numpy as np
from pragmatic_rc import neural
from pragmatic_rc.envs import GridWorld

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
H, UPDATES = 40, 400
BETAS = [0.02, 0.05, 0.1, 0.2]
g = GridWorld(n=N, slip=0.2, gamma=0.9)
mdp = g.to_mdp()

def always_value(beta):
    V = np.zeros(mdp.n_states)
    for _ in range(H):
        V = (mdp.Rbar + mdp.gamma * mdp.P @ V).max(axis=0)
    return V.mean() - beta * sum(mdp.gamma**t for t in range(1, H + 1))

res = {}
for beta in BETAS:
    pol = neural.train(g, beta, updates=UPDATES, horizon=H)
    learned = neural.evaluate(g, pol, beta, horizon=H)
    nev = neural.train(g, beta, updates=UPDATES, horizon=H, force_send=0)
    never = neural.evaluate(g, nev, beta, horizon=H, force_send=0)
    res[beta] = dict(learned=learned, never=never, always=always_value(beta))
    print(beta, res[beta], flush=True)
json.dump(res, open(f"figures/large_{N}.json", "w"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
b = BETAS
ax[0].errorbar(b, [res[x]["learned"][0] for x in b], yerr=[1.96 * res[x]["learned"][1] for x in b], marker="o", color="#2a7f62", label="learned")
ax[0].plot(b, [res[x]["never"][0] for x in b], marker="s", color="#999", label="never transmit")
ax[0].plot(b, [res[x]["always"] for x in b], marker="^", color="#d9822b", label="always transmit")
ax[0].set_xlabel("cost per transmission β"); ax[0].set_ylabel("net value"); ax[0].legend(fontsize=8)
ax[0].set_title(f"{N}×{N} GridWorld ({N*N} states)", fontsize=10)
ax[1].plot(b, [res[x]["learned"][2] for x in b], marker="o", color="#2a7f62")
ax[1].set_xlabel("cost per transmission β"); ax[1].set_ylabel("learned: transmissions per step"); ax[1].set_ylim(0, 1)
fig.tight_layout(); fig.savefig(f"figures/large_{N}.png", dpi=160)
