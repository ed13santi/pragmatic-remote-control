"""Plot score against transmissions for the jump experiments (reads figures/jumps_*.json)."""
import glob, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = [json.load(open(f)) for f in glob.glob("figures/jumps_*.json")]
tasks = sorted({r["task"] for r in runs})
titles = {"inverted_pendulum_kicks": "Random impulse kicks", "inverted_pendulum_switching": "Hidden actuator sign flips"}
fig, axes = plt.subplots(1, len(tasks), figsize=(4.8 * len(tasks), 3.8), squeeze=False)
for ax, task in zip(axes[0], tasks):
    rs = [r for r in runs if r["task"] == task]
    th = sorted([r for r in rs if r["kind"] == "threshold"], key=lambda r: r["value"])
    ax.errorbar([r["sends"] for r in th], [r["score"] for r in th], yerr=[1.96 * r["se"] for r in th],
                marker="s", color="#777", capsize=3, label="threshold (swept θ)")
    for r in th:
        ax.annotate(f"θ={r['value']}", (r["sends"], r["score"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for kind, color, label in [("joint", "#2a7f62", "learned"), ("joint_rich", "#1f3b73", "learned, richer belief"),
                               ("always", "#d9822b", "always send")]:
        for r in rs:
            if r["kind"] == kind:
                ax.errorbar(r["sends"], r["score"], yerr=1.96 * r["se"], marker="o", color=color, capsize=3, label=label)
    ax.set_title(titles.get(task, task), fontsize=10)
    ax.set_xlabel("transmissions per episode"); ax.set_ylabel("score (steps survived − β · sends)")
axes[0][0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/jumps.png", dpi=160)
