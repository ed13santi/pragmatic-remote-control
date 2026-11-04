"""Reward and channel use as a function of density d and communication cost beta.

Reproduces the structure of Figs. 2 and 3 of the paper for the remote control
and the remote estimation tasks (MPI and API, 30 states).

    python experiments/grid.py control
    python experiments/grid.py estimation
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from common import FIGURES, heatmap, metrics, run_api, run_mpi

from pragmatic_rc import random_control_mdp, random_estimation_mdp

N_STATES, T_MAX, GAMMA = 30, 8, 0.9
# The two tasks have different reward scales, so the cost range differs.
BETA_RANGE = {"control": (0.0, 0.5, 11), "estimation": (0.0, 2.0, 9)}
DENSITIES = np.linspace(0.1, 0.9, 9)


def build(task, density, seed):
    rng = np.random.default_rng(seed)
    if task == "control":
        return random_control_mdp(N_STATES, 3, density, GAMMA, rng)
    return random_estimation_mdp(N_STATES, density, GAMMA, rng)


def main(task: str) -> None:
    BETAS = np.linspace(*BETA_RANGE[task])
    shape = (len(DENSITIES), len(BETAS))
    out = {k: np.zeros(shape) for k in ("mpi_r", "mpi_c", "api_r", "api_c", "gap")}
    for i, d in enumerate(DENSITIES):
        mdp = build(task, d, seed=i)  # one MDP per density, shared across beta
        for j, beta in enumerate(BETAS):
            r, c = metrics(mdp, run_mpi(mdp, beta, T_MAX))
            out["mpi_r"][i, j], out["mpi_c"][i, j] = r, c
            runs = [run_api(mdp, beta, T_MAX, init)[1] for init in ("never", "always")]
            vals = [ev.value(beta).mean() for ev in runs]
            best = runs[int(np.argmax(vals))]
            out["api_r"][i, j], out["api_c"][i, j] = metrics(mdp, best)
            out["gap"][i, j] = abs(vals[0] - vals[1])
        print(f"density {d:.2f} done", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), constrained_layout=True)
    panels = [
        ("mpi_r", "Reward (MPI, pull)"),
        ("mpi_c", "Channel use (MPI, pull)"),
        ("gap", "API: gap between the two starts"),
        ("api_r", "Reward (API, best of two starts)"),
        ("api_c", "Channel use (API, best of two starts)"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        vmax = 1.0 if key.endswith("_c") else None
        im = heatmap(ax, out[key], BETAS, DENSITIES, title, vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax)
    axes.flat[-1].axis("off")
    fig.suptitle(f"Remote {task}, {N_STATES} states, discount {GAMMA}, Tmax {T_MAX}")
    path = FIGURES / f"grid_{task}.png"
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "control")
