"""Sanity checks: (1) float32-vs-float64 inertia; (2) cost() vs brute force; (3) exact reproducibility of seeded runs."""
import json, time
import numpy as np
from sklearn.cluster import KMeans
from common import build_data, OUT, countsketch
import importlib.util, sys

D, y, cols = build_data(); n, d = D.shape
r1 = json.load(open(f"{OUT}/q1_results.json")); r2 = json.load(open(f"{OUT}/q2_results.json"))

def cost(X, B, chunk=50000):
    B = np.asarray(B, dtype=np.float64); bb = (B ** 2).sum(1); tot = 0.0
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk].astype(np.float64)
        tot += np.maximum(((x ** 2).sum(1)[:, None] - 2 * x @ B.T + bb[None, :]).min(1), 0).sum()
    return tot

# (1) k-means in float64: does sklearn's inertia_ then agree with our cost?
km = KMeans(n_clusters=15, n_init=5, random_state=2026).fit(D.astype(np.float64))
print("float64 KMeans: inertia_ =", km.inertia_, " our cost =", cost(D, km.cluster_centers_))
km32 = KMeans(n_clusters=15, n_init=5, random_state=2026).fit(D)
print("float32 KMeans: inertia_ =", km32.inertia_, " our cost =", cost(D, km32.cluster_centers_), " saved:", r1["a"]["cost"])

# (2) brute-force check of cost() on 3000 rows
rng = np.random.default_rng(0); idx = rng.choice(n, 3000, replace=False); X = D[idx].astype(np.float64)
B = km32.cluster_centers_
bf = sum(min(((X[i] - b) ** 2).sum() for b in B) for i in range(len(X)))
print("cost() vs brute force on 3000 rows:", cost(D[idx], B), bf)

# (3) reproducibility: redo Q1 m=10 rep 0 and Q2 m=500 rep 0 and compare to saved numbers
m, r = 10, 0
rng = np.random.default_rng([2026, 1, m, r]); A = rng.normal(0.0, 1.0 / np.sqrt(m), size=(m, d))
E = D @ A.T.astype(np.float32); k = KMeans(n_clusters=15, n_init=5, random_state=2026 + 100 * m + r).fit(E)
lab = k.labels_; Bh = np.array([D[lab == j].astype(np.float64).mean(0) for j in range(15)])
print("Q1 m=10 rep0 cost:", cost(D, Bh), "saved:", r1["b"]["10"]["cost"][0])
D64 = D.astype(np.float64); m, r = 500, 0
rng = np.random.default_rng([2026, 1, m, r]); SD, Sy = countsketch(D64, y, m, rng)
at = np.linalg.lstsq(SD, Sy, rcond=None)[0]
print("Q2 m=500 rep0 loss:", float(((D64 @ at - y) ** 2).sum()), "saved:", r2["b"]["500"]["loss"][0])
