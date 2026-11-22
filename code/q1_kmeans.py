"""Q1: JL for k-means on KDD Cup 99.  Run: python q1_kmeans.py"""
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from common import build_data, OUT, Timer

K = 15
MS = [5, 10, 15, 20, 25]
REPS = 5
SEED = 2026

D, y, cols = build_data()
n, d = D.shape
print(f"D: n={n}, d={d}", flush=True)


def cost(X, B, chunk=50000):
    """cost(X,B): sum over rows of the squared distance to the nearest centre."""
    B = np.asarray(B, dtype=np.float64)
    bb = (B ** 2).sum(1)
    tot = 0.0
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk].astype(np.float64)
        d2 = (x ** 2).sum(1)[:, None] - 2 * x @ B.T + bb[None, :]
        tot += np.maximum(d2.min(1), 0).sum()
    return tot


def centroids(X, labels, k=K):
    """Cluster means mu_j(X)."""
    cnt = np.bincount(labels, minlength=k).astype(np.float64)
    out = np.empty((k, X.shape[1]))
    for j in range(X.shape[1]):
        out[:, j] = np.bincount(labels, weights=X[:, j].astype(np.float64), minlength=k)
    return out / cnt[:, None]


def cost_partition(X, labels, k=K):
    """cost_X(C) for a partition C (given as labels)."""
    mu = centroids(X, labels, k)
    tot = 0.0
    for s in range(0, X.shape[0], 100000):
        r = X[s:s + 100000].astype(np.float64) - mu[labels[s:s + 100000]]
        tot += (r ** 2).sum()
    return tot


def kmeans(X, seed):
    return KMeans(n_clusters=K, n_init=5, random_state=seed).fit(X)


res = {"n": n, "d": d}

# (a)
with Timer() as t:
    km = kmeans(D, SEED)
B = km.cluster_centers_
labels_D = km.labels_
costDB = cost(D, B)
res["a"] = {"cost": costDB, "inertia": float(km.inertia_), "time": t.dt}
print(f"(a) cost(D,B)={costDB:.4f} (inertia {km.inertia_:.4f}) time={t.dt:.2f}s", flush=True)
cost_D_CD = cost_partition(D, labels_D)  # needed in (e)
print("   cost_D(C_D) =", cost_D_CD, flush=True)

# (b),(c),(d)
rb = {m: {"ratio": [], "ratio_pinv": [], "cost": [], "cost_pinv": [], "time": [], "time_A": [], "time_total": []}
      for m in MS}
for m in MS:
    for r in range(REPS):
        rng = np.random.default_rng([SEED, 1, m, r])
        t0 = time.perf_counter()
        A = rng.normal(0.0, 1.0 / np.sqrt(m), size=(m, d))
        tA = time.perf_counter() - t0
        t0 = time.perf_counter()
        E = D @ A.T.astype(np.float32)
        tE = time.perf_counter() - t0
        t0 = time.perf_counter()
        kme = kmeans(E, SEED + 100 * m + r)
        tK = time.perf_counter() - t0
        Bhat = centroids(D, kme.labels_)  # centres in R^d
        Bplus = (np.linalg.pinv(A) @ kme.cluster_centers_.T.astype(np.float64)).T
        c_hat, c_plus = cost(D, Bhat), cost(D, Bplus)
        rb[m]["cost"].append(c_hat); rb[m]["cost_pinv"].append(c_plus)
        rb[m]["ratio"].append(c_hat / costDB); rb[m]["ratio_pinv"].append(c_plus / costDB)
        rb[m]["time"].append(tE + tK)   # time for (d)
        rb[m]["time_A"].append(tA)
        rb[m]["time_total"].append(tA + tE + tK)
        print(f"m={m:2d} rep={r}: cost={c_hat:.2f} ratio={c_hat/costDB:.4f} "
              f"ratio_pinv={c_plus/costDB:.3f} time(E+kmeans)={tE+tK:.2f}s", flush=True)
res["b"] = {str(m): v for m, v in rb.items()}

# (e)
rng = np.random.default_rng([SEED, 2])
I, L = [], []
while len(I) < 10 ** 4:
    i, l = rng.integers(0, n, size=2)
    if i != l and np.any(D[i] != D[l]):
        I.append(i); L.append(l)
I, L = np.array(I), np.array(L)
diff = D[I].astype(np.float64) - D[L].astype(np.float64)
den = (diff ** 2).sum(1)
assert (den > 0).all()
pair_ratio, cost_ratio = {}, {}
for m in MS:
    rng = np.random.default_rng([SEED, 3, m])
    A = rng.normal(0, 1 / np.sqrt(m), size=(m, d))
    pair_ratio[m] = (((diff @ A.T) ** 2).sum(1) / den)
    cr = []
    for r in range(20):
        A = rng.normal(0, 1 / np.sqrt(m), size=(m, d))
        E = (D @ A.T.astype(np.float32))
        cr.append(cost_partition(E, labels_D) / cost_D_CD)
    cost_ratio[m] = np.array(cr)
    print(f"(e) m={m}: pair ratio mean={pair_ratio[m].mean():.3f} sd={pair_ratio[m].std():.3f} | "
          f"cost ratio mean={cost_ratio[m].mean():.3f} sd={cost_ratio[m].std():.3f}", flush=True)
res["e"] = {str(m): {"pair_mean": float(pair_ratio[m].mean()), "pair_sd": float(pair_ratio[m].std()),
                     "pair_q": [float(q) for q in np.percentile(pair_ratio[m], [0, 25, 50, 75, 100])],
                     "cost_mean": float(cost_ratio[m].mean()), "cost_sd": float(cost_ratio[m].std()),
                     "cost_q": [float(q) for q in np.percentile(cost_ratio[m], [0, 25, 50, 75, 100])]}
            for m in MS}
res["cost_D_CD"] = cost_D_CD
json.dump(res, open(f"{OUT}/q1_results.json", "w"), indent=1)

# figures
x = np.arange(len(MS))
mean = np.array([np.mean(rb[m]["ratio"]) for m in MS])
lo = np.array([np.min(rb[m]["ratio"]) for m in MS])
hi = np.array([np.max(rb[m]["ratio"]) for m in MS])
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(x, mean, color="#4c78a8", yerr=[mean - lo, hi - mean], capsize=5, label="mean (bars: min-max of 5 runs)")
for j, m in enumerate(MS):
    ax.scatter([j] * REPS, rb[m]["ratio"], color="k", s=10, zorder=3)
ax.axhline(1, color="r", ls="--", lw=1, label="ratio = 1")
ax.set_xticks(x); ax.set_xticklabels(MS)
ax.set_xlabel("m"); ax.set_ylabel(r"cost$(D,\hat B)$ / cost$(D,B)$")
ax.set_title(r"Q1(b): cost ratio of $\hat B$ (clusters from JL-projected data)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/q1b_ratio.png", dpi=150); plt.close(fig)

# (c)
w = 0.38
mp = np.array([np.mean(rb[m]["ratio_pinv"]) for m in MS])
lop = np.array([np.min(rb[m]["ratio_pinv"]) for m in MS])
hip = np.array([np.max(rb[m]["ratio_pinv"]) for m in MS])
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(x - w / 2, mean, w, yerr=[mean - lo, hi - mean], capsize=4, label=r"$\hat B$ (cluster means)", color="#4c78a8")
ax.bar(x + w / 2, mp, w, yerr=[mp - lop, hip - mp], capsize=4, label=r"$B^+=A^+a$ (pseudo-inverse)", color="#f58518")
ax.axhline(1, color="r", ls="--", lw=1)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(MS)
ax.set_xlabel("m"); ax.set_ylabel("cost ratio w.r.t. cost(D,B)  (log scale)")
ax.set_title(r"Q1(c): $\hat B$ vs $B^+$"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/q1c_ratio.png", dpi=150); plt.close(fig)

# (d)
tm = np.array([np.mean(rb[m]["time"]) for m in MS])
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(MS, tm, "o-", label="compute E + k-means on E (mean of 5)")
ax.axhline(res["a"]["time"], color="r", ls="--", label=f"k-means on D ({res['a']['time']:.1f}s)")
ax.set_xlabel("m"); ax.set_ylabel("wall-clock time (s)"); ax.set_title("Q1(d): running time")
ax.set_ylim(bottom=0); ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/q1d_time.png", dpi=150); plt.close(fig)

# (e)
fig, ax = plt.subplots(figsize=(8, 4.5))
p1 = ax.boxplot([pair_ratio[m] for m in MS], positions=np.arange(len(MS)) - 0.18, widths=0.3, patch_artist=True,
                showfliers=True, flierprops=dict(markersize=2, alpha=0.3))
p2 = ax.boxplot([cost_ratio[m] for m in MS], positions=np.arange(len(MS)) + 0.18, widths=0.3, patch_artist=True,
                flierprops=dict(markersize=3))
for b in p1["boxes"]: b.set_facecolor("#9ecae9")
for b in p2["boxes"]: b.set_facecolor("#f5b27a")
ax.axhline(1, color="r", ls="--", lw=1)
ax.set_xticks(np.arange(len(MS))); ax.set_xticklabels(MS)
ax.set_xlabel("m"); ax.set_ylabel("ratio")
ax.legend([p1["boxes"][0], p2["boxes"][0]],
          [r"(i) $\|E_i-E_l\|^2/\|D_i-D_l\|^2$, $10^4$ pairs, one $A$", r"(ii) cost$_E(C_D)$/cost$_D(C_D)$, 20 draws of $A$"])
ax.set_title("Q1(e): distortion of pairwise distances vs. of the k-means cost")
fig.tight_layout(); fig.savefig(f"{OUT}/q1e_box.png", dpi=150); plt.close(fig)
print("done", flush=True)
