"""How often does a single run of MPI reach the global optimum?

Compares MPI against exhaustive search over open-loop plans on small random
MDPs. Prints the fraction of random initial policies that reach the optimum.
"""

from __future__ import annotations

import numpy as np

from pragmatic_rc import pull, random_control_mdp


def main(n_mdps: int = 20, starts: int = 20, t_max: int = 4) -> None:
    hits, total = 0, 0
    solved_by_restarts = 0
    for seed in range(n_mdps):
        for beta in (0.1, 0.5):
            mdp = random_control_mdp(4, 2, 0.5, 0.9, np.random.default_rng(seed))
            opt = pull.brute_force_value(mdp, beta, t_max)
            rng = np.random.default_rng(1000 + seed)
            reached = []
            for _ in range(starts):
                _, ev = pull.mpi(mdp, beta, t_max, rng=rng)
                reached.append(np.max(np.abs(ev.value(beta) - opt)) < 1e-8)
            hits += sum(reached)
            total += starts
            solved_by_restarts += any(reached)
    print(f"single start reaches the optimum in {hits / total:.0%} of runs")
    print(f"best of {starts} starts reaches it in {solved_by_restarts} of {2 * n_mdps} instances")


if __name__ == "__main__":
    main()
