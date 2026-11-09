"""Value iteration for the error MDP W(e)=min{beta+g*E W(w), e'Me+g*E W(Ae+w)} on a 2-D grid.

Checks whether the send region is an M-ball (threshold structure). Prints, per dynamics matrix, the M-norm of the nearest
sending point and of the farthest silent point; a ratio of 1 means an exact M-ball threshold."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
def solve(A, M, sigma=0.5, beta=3.0, g=0.9, L=6.0, n=161, iters=400):
    ax = np.linspace(-L, L, n); dx = ax[1]-ax[0]
    X, Y = np.meshgrid(ax, ax, indexing="ij"); E = np.stack([X, Y], -1)
    d = np.einsum("...i,ij,...j->...", E, M, E)
    AE = E @ A.T
    idx = np.stack([(AE[..., 0]+L)/dx, (AE[..., 1]+L)/dx])
    W = np.zeros((n, n))
    for _ in range(iters):
        S = gaussian_filter(W, sigma/dx, mode="nearest")
        send = beta + g*S[n//2, n//2]
        stay = d + g*map_coordinates(S, idx, order=1, mode="nearest")
        Wn = np.minimum(send, stay)
        if np.abs(Wn-W).max() < 1e-9: W = Wn; break
        W = Wn
    return ax, d, stay, send, W
def report(name, A, M):
    ax, d, stay, send, W = solve(A, M)
    region = stay >= send                      # send region
    n = len(ax); X, Y = np.meshgrid(ax, ax, indexing="ij")
    r = np.sqrt(np.einsum("...i,ij,...j->...", np.stack([X, Y], -1), M, np.stack([X, Y], -1)))
    inner = r[region].min(); outer = r[~region].max()
    print(f"{name}: M-norm of nearest sending point {inner:.3f}, farthest silent point {outer:.3f}, "
          f"ratio outer/inner {outer/inner:.3f}  (1.0 = exact M-ball threshold)")
th = 0.7
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
I = np.eye(2)
report("A = 1.1*rotation, M=I (assumption holds)", 1.1*R, I)
report("A = 0.9*rotation, M=I (assumption holds)", 0.9*R, I)
report("A = [[0.9,0.9],[0,0.9]] non-normal, M=I (assumption fails)", np.array([[0.9,0.9],[0,0.9]]), I)
report("A = [[1.0,1.5],[0,1.0]] strongly non-normal, M=I", np.array([[1.0,1.5],[0,1.0]]), I)
