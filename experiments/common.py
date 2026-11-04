"""Shared helpers for the experiment scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pragmatic_rc import pull, push

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def run_mpi(mdp, beta, t_max, restarts=8, seed=0):
    return pull.mpi(mdp, beta, t_max, restarts=restarts, rng=np.random.default_rng(seed))[1]


def run_api(mdp, beta, t_max, init, seed=0):
    c0 = (push.never_transmit if init == "never" else push.always_transmit)(t_max, mdp.n_states)
    return push.api(mdp, beta, t_max, c0, rng=np.random.default_rng(seed))


def metrics(mdp, ev):
    """Mean discounted reward and average channel use per step (in [0, 1])."""
    return ev.reward.mean(), (1 - mdp.gamma) * ev.tx.mean()


def heatmap(ax, data, betas, densities, title, vmin=None, vmax=None):
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[betas[0], betas[-1], densities[0], densities[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel(r"communication cost $\beta$")
    ax.set_ylabel("transition density $d$")
    return im
