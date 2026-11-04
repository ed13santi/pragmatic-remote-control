"""Optimal (HSVI) against alternating (API) and pull-based (MPI) policies.

Plots the expected discounted return net of transmission cost, as a function of
the communication cost, for small remote control and remote estimation tasks.
The optimal curve is a certified interval (lower and upper bound).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import FIGURES

from pragmatic_rc import optimal, pull, push, random_control_mdp, random_estimation_mdp

N_STATES, T_MAX, GAMMA, EPS = 4, 10, 0.9, 0.01


def methods(mdp, beta):
    n = mdp.n_states
    mpi = pull.mpi(mdp, beta, T_MAX, restarts=20)[1].value(beta).mean()
    api = max(
        push.api(mdp, beta, T_MAX, c0)[1].value(beta).mean()
        for c0 in (push.never_transmit(T_MAX, n), push.always_transmit(T_MAX, n))
    )
    lo, hi = optimal.solve(mdp, beta, eps=EPS).value_bounds
    return mpi, api, lo, hi


def main() -> None:
    # Control instances where transmitting is never worthwhile say little, so we
    # use a noisier 6-state task in which the optimal policy does transmit.
    tasks = {
        "Remote control": (random_control_mdp(6, 2, 0.3, GAMMA, np.random.default_rng(1)), np.linspace(0.02, 0.4, 12)),
        "Remote estimation": (
            random_estimation_mdp(N_STATES, 0.5, GAMMA, np.random.default_rng(1)),
            np.linspace(0.1, 2.0, 12),
        ),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, (title, (mdp, betas)) in zip(axes, tasks.items()):
        rows = np.array([methods(mdp, b) for b in betas])
        ax.fill_between(betas, rows[:, 2], rows[:, 3], color="C2", alpha=0.35, label="optimal (HSVI bounds)")
        ax.plot(betas, rows[:, 1], "^-", color="C1", ms=4, label="alternating (API)")
        ax.plot(betas, rows[:, 0], "o-", color="C0", ms=4, label="pull-based (MPI)")
        ax.set_title(f"{title}, {mdp.n_states} states")
        ax.set_xlabel(r"communication cost $\beta$")
        ax.set_ylabel("discounted return net of cost")
        print(title, "max gap optimal - alternating: %.3f" % np.max(rows[:, 2] - rows[:, 1]))
    axes[0].legend(frameon=False, fontsize=8)
    path = FIGURES / "compare.png"
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    main()
