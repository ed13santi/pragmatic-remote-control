"""Plot the small-problem validation of the sampling-based and learned planners.

Numbers come from the runs described in the README: the exact HSVI bounds
(`optimal.solve`), POMCP with `AllSubsets` (150 simulations per decision, 20
episodes) and the actor-critic policies of `pragmatic_rc.neural` (300 updates,
2000 evaluation episodes). Each is a net value: reward minus transmission cost.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = {
    "Estimation, 3 states (β=0.3)": dict(bounds=(4.381, 4.425), pomcp=(4.407, 0.033), neural=(4.383, 0.004)),
    "Control, 3 states (β=0.1)": dict(bounds=(3.796, 3.842), pomcp=(3.723, 0.027), neural=(3.783, 0.003)),
}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, (title, d) in zip(axes, data.items()):
    lo, hi = d["bounds"]
    ax.axhspan(lo, hi, color="#cfe3f5", label="exact bounds (HSVI)")
    for i, (name, c) in enumerate([("POMCP", "#d9822b"), ("Learned (actor-critic)", "#2a7f62")]):
        m, se = d["pomcp" if i == 0 else "neural"]
        ax.errorbar(i, m, yerr=1.96 * se, fmt="o", color=c, capsize=4, label=name)
    ax.set_xlim(-0.6, 1.6); ax.set_xticks([])
    ax.set_title(title, fontsize=10); ax.set_ylabel("net value")
    ax.set_ylim(lo - 0.25, hi + 0.12)
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("figures/validation.png", dpi=160)
