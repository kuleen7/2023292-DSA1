"""Redraws the Q1(b) and Q2(b) figures (clearer versions) from the saved JSON results (no recomputation)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT

r1 = json.load(open(f"{OUT}/q1_results.json"))
r2 = json.load(open(f"{OUT}/q2_results.json"))

# ---- Q1(b): mean with min-max whiskers + individual runs
MS = [5, 10, 15, 20, 25]
fig, ax = plt.subplots(figsize=(6.8, 4.2))
x = np.arange(len(MS))
for j, m in enumerate(MS):
    v = np.array(r1["b"][str(m)]["ratio"])
    ax.bar(j, v.mean() - 0.9, bottom=0.9, color="#9ecae9", width=0.55, edgecolor="#4c78a8")
    ax.vlines(j, v.min(), v.max(), color="k", lw=1.5)
    ax.scatter([j] * len(v), v, color="k", s=14, zorder=3)
    ax.text(j + 0.32, v.mean(), f"{v.mean():.3f}", va="center", fontsize=8)
ax.axhline(1, color="r", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(MS); ax.set_ylim(0.9, 1.25)
ax.set_xlabel("m"); ax.set_ylabel(r"cost$(D,\hat B)$ / cost$(D,B)$")
ax.set_title(r"Q1(b): bars = mean, line = min-max, dots = the 5 runs (y-axis starts at 0.9)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/q1b_ratio.png", dpi=150); plt.close(fig)

# ---- Q2(b): ratio (log-log) and sketched loss relative to Loss(D,y,b)
M2 = [50, 100, 200, 500, 1000, 2000, 5000]
rank = r2["rank"]; Lb = r2["a"]["loss"]
mean = np.array([np.mean(r2["b"][str(m)]["ratio"]) for m in M2])
lo = np.array([np.min(r2["b"][str(m)]["ratio"]) for m in M2])
hi = np.array([np.max(r2["b"][str(m)]["ratio"]) for m in M2])
sk = np.array([np.mean(r2["b"][str(m)]["sk_loss"]) for m in M2]) / Lb
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].errorbar(M2, mean, yerr=[mean - lo, hi - mean], fmt="o-", capsize=4)
ax[0].axhline(1, color="r", ls="--", lw=1, label="ratio = 1")
ax[0].axvline(rank, color="gray", ls=":", label=f"m = rank(D) = {rank}")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("m (log)"); ax[0].set_ylabel(r"Loss$(D,y,\tilde a)$ / Loss$(D,y,b)$  (log)")
ax[0].set_title("Q2(b): ratio, mean and min/max of 5 runs"); ax[0].legend()
ax[1].plot(M2, sk, "s-", color="#f58518", label=r"mean $\|SD\tilde a-Sy\|^2$ / Loss$(D,y,b)$")
ax[1].plot(M2, [max(0, 1 - rank / m) for m in M2], "k:", label=r"$1-\mathrm{rank}/m$ (reference)")
ax[1].axhline(1, color="r", ls="--", lw=1)
ax[1].axvline(rank, color="gray", ls=":")
ax[1].set_xscale("log"); ax[1].set_xlabel("m (log)"); ax[1].set_ylabel("sketched loss / Loss(D,y,b)")
ax[1].set_title("Q2(b): sketched loss (exactly 0 for m < rank)"); ax[1].legend(loc="center right")
fig.tight_layout(); fig.savefig(f"{OUT}/q2b_ratio.png", dpi=150); plt.close(fig)

# ---- Q1(d): time vs m with clean integer ticks
tm = [np.mean(r1["b"][str(m)]["time"]) for m in MS]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(MS, tm, "o-", label="compute E + k-means on E (mean of 5 runs)")
ax.axhline(r1["a"]["time"], color="r", ls="--", label=f"k-means on D, part (a) ({r1['a']['time']:.1f} s)")
ax.set_xticks(MS); ax.set_xlabel("m"); ax.set_ylabel("wall-clock time (s)"); ax.set_ylim(bottom=0)
ax.set_title("Q1(d): running time"); ax.legend(loc="center right"); fig.tight_layout()
fig.savefig(f"{OUT}/q1d_time.png", dpi=150); plt.close(fig)
