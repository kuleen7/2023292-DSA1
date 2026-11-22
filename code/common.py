"""Data matrix construction and CountSketch helpers."""
import os
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_HOME = os.path.join(HERE, "..", "data")
CACHE = os.path.join(DATA_HOME, "D_cache.npz")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

CAT = ["protocol_type", "service", "flag"]


def build_data(verbose=True):
    """Returns D (float32, scaled to [0,1]), y (0/1 attack label) and the column names."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["D"], z["y"], list(z["cols"])
    from sklearn.datasets import fetch_kddcup99
    kdd = fetch_kddcup99(subset=None, as_frame=True, data_home=DATA_HOME)
    X = kdd.data.copy()
    # (i) one-hot encode the text features, other 38 to float
    for c in CAT:
        X[c] = X[c].map(lambda b: b.decode() if isinstance(b, bytes) else str(b))
    num_cols = [c for c in X.columns if c not in CAT]
    Xnum = X[num_cols].astype(float)
    Xcat = pd.get_dummies(X[CAT], dtype=float)
    M = pd.concat([Xnum, Xcat], axis=1)
    if verbose:
        print("after one-hot:", M.shape)
    # (ii) drop constant columns
    keep = M.nunique() > 1
    if verbose:
        print("constant columns dropped:", list(M.columns[~keep]))
    M = M.loc[:, keep]
    # (iii) min-max scaling
    mn, mx = M.min(), M.max()
    M = (M - mn) / (mx - mn)
    D = M.to_numpy(dtype=np.float32)
    y = (kdd.target != b"normal.").to_numpy(dtype=float)
    cols = list(M.columns)
    np.savez(CACHE, D=D, y=y, cols=np.array(cols, dtype=object))
    return D, y, cols


def countsketch(D, y, m, rng):
    """CountSketch of (D, y): row i is added with sign sigma(i) to row h(i). S is never stored.
    np.bincount per column gives the same result as np.add.at(SD, h, sigma[:, None] * D) but is much faster."""
    n, d = D.shape
    h = rng.integers(0, m, size=n)               # bucket h(i)
    sg = rng.integers(0, 2, size=n) * 2.0 - 1.0  # sign sigma(i)
    SD = np.empty((m, d), dtype=np.float64)
    for j in range(d):
        SD[:, j] = np.bincount(h, weights=sg * D[:, j], minlength=m)
    Sy = np.bincount(h, weights=sg * y, minlength=m)
    return SD, Sy


def countsketch_vec(v, m, rng):
    """CountSketch of one vector (Q2(d))."""
    n = v.shape[0]
    h = rng.integers(0, m, size=n)
    sg = rng.integers(0, 2, size=n) * 2.0 - 1.0
    return np.bincount(h, weights=sg * v, minlength=m)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
