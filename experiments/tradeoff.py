"""Reward versus channel use for pull (MPI) and push (API from two starts).

Counterpart of Fig. 4 of the paper (10 states, density 0.9). The JPO scheme is
not implemented, so the curve of the best push policy is the upper envelope of
the two API runs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import FIGURES, metrics, run_api, run_mpi

from pragmatic_rc import random_control_mdp

N_STATES, T_MAX, GAMMA, DENSITY = 10, 8, 0.9, 0.9


def main() -> None:
    mdp = random_control_mdp(N_STATES, 3, DENSITY, GAMMA, np.random.default_rng(0))
    betas = np.linspace(0.0, 0.4, 41)
    curves = {"MPI (pull)": [], "API from never": [], "API from always": []}
    for beta in betas:
        curves["MPI (pull)"].append(metrics(mdp, run_mpi(mdp, beta, T_MAX)))
        curves["API from never"].append(metrics(mdp, run_api(mdp, beta, T_MAX, "never")[1]))
        curves["API from always"].append(metrics(mdp, run_api(mdp, beta, T_MAX, "always")[1]))

    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    for (label, pts), marker in zip(curves.items(), "o^s"):
        pts = np.array(pts)
        order = np.argsort(pts[:, 1])
        ax.plot(pts[order, 1], pts[order, 0], marker=marker, ms=4, lw=1.2, label=label)
    ax.set_xlabel("average channel use")
    ax.set_ylabel("discounted reward")
    ax.set_title(f"Trade-off, {N_STATES} states, d = {DENSITY}")
    ax.legend(frameon=False)
    path = FIGURES / "tradeoff.png"
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    main()
