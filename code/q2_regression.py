import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import build_data, OUT, countsketch, countsketch_vec

SEED = 2026
MS = [50, 100, 200, 500, 1000, 2000, 5000]
REPS = 5

D32, y, cols = build_data()
D = D32.astype(np.float64)
n, d = D.shape
print(f"D: n={n}, d={d}; attack fraction y={y.mean():.4f}", flush=True)


def lstsq(X, t):
    return np.linalg.lstsq(X, t, rcond=None)[0]


def loss(X, t, a):
    r = X @ a - t
    return float(r @ r)


res = {"n": n, "d": d}

sv = np.linalg.svd(D, compute_uv=False)
rank = int(np.linalg.matrix_rank(D))
res["rank"] = rank
res["sv_tail"] = [float(s) for s in sv[-6:]]
res["sv_max"] = float(sv[0])
print("rank(D) =", rank, "smallest singular values:", sv[-6:], flush=True)

times = []
for _ in range(3):
    t0 = time.perf_counter()
    b = lstsq(D, y)
    times.append(time.perf_counter() - t0)
Lb = loss(D, y, b)
res["a"] = {"loss": Lb, "time_mean": float(np.mean(times)), "times": times}
print(f"(a) Loss(D,y,b)={Lb:.4f}  time={np.mean(times):.3f}s {times}", flush=True)

rb = {}
for m in MS:
    rows = {"loss": [], "ratio": [], "sk_loss": [], "time": []}
    for r in range(REPS):
        rng = np.random.default_rng([SEED, 1, m, r])
        t0 = time.perf_counter()
        SD, Sy = countsketch(D, y, m, rng)
        at = lstsq(SD, Sy)
        dt = time.perf_counter() - t0
        L = loss(D, y, at)
        rows["loss"].append(L); rows["ratio"].append(L / Lb)
        rows["sk_loss"].append(float(((SD @ at - Sy) ** 2).sum())); rows["time"].append(dt)
        print(f"m={m:5d} rep={r}: Loss={L:.2f} ratio={L/Lb:.4f} sketched={rows['sk_loss'][-1]:.2f} time={dt:.3f}s",
              flush=True)
    rb[m] = rows
res["b"] = {str(m): v for m, v in rb.items()}

v = D @ b - y
nv2 = float(v @ v)
v4 = float((v ** 4).sum())
rd = {}
for m in [50, 500, 5000]:
    rr = []
    for r in range(100):
        rng = np.random.default_rng([SEED, 4, m, r])
        Sv = countsketch_vec(v, m, rng)
        rr.append(float(Sv @ Sv) / nv2)
    rr = np.array(rr)
    theo = float(np.sqrt(2.0 / m * (1 - v4 / nv2 ** 2)))
    rd[m] = {"vals": rr.tolist(), "mean": float(rr.mean()), "sd": float(rr.std(ddof=1)), "theory_sd": theo}
    print(f"(d) m={m}: mean={rr.mean():.4f} sd={rr.std(ddof=1):.4f} (theory sd {theo:.4f})", flush=True)
res["d"] = {str(m): v_ for m, v_ in rd.items()}

rng = np.random.default_rng(SEED + 5)
idx = rng.choice(n, size=50000, replace=False)
Dp, yp = D[idx], y[idx]
bp = lstsq(Dp, yp)
Lbp = loss(Dp, yp, bp)
print(f"(e) Loss(D',y',b')={Lbp:.4f}", flush=True)
m = 500
re = {"count": {"t_form": [], "t_solve": [], "ratio": []}, "dense": {"t_form": [], "t_solve": [], "ratio": []}}
for r in range(REPS):
    rng = np.random.default_rng([SEED, 6, r])
    t0 = time.perf_counter()
    SD, Sy = countsketch(Dp, yp, m, rng)
    t1 = time.perf_counter()
    at = lstsq(SD, Sy)
    t2 = time.perf_counter()
    re["count"]["t_form"].append(t1 - t0); re["count"]["t_solve"].append(t2 - t1)
    re["count"]["ratio"].append(loss(Dp, yp, at) / Lbp)
    rng = np.random.default_rng([SEED, 7, r])
    t0 = time.perf_counter()
    S = rng.normal(0.0, 1.0 / np.sqrt(m), size=(m, Dp.shape[0]))
    SD, Sy = S @ Dp, S @ yp
    t1 = time.perf_counter()
    at = lstsq(SD, Sy)
    t2 = time.perf_counter()
    re["dense"]["t_form"].append(t1 - t0); re["dense"]["t_solve"].append(t2 - t1)
    re["dense"]["ratio"].append(loss(Dp, yp, at) / Lbp)
    del S
    print(f"  rep {r}: count form={re['count']['t_form'][-1]:.3f}s ratio={re['count']['ratio'][-1]:.5f} | "
          f"dense form={re['dense']['t_form'][-1]:.3f}s ratio={re['dense']['ratio'][-1]:.5f}", flush=True)
res["e"] = {"loss_bp": Lbp, **re}
json.dump(res, open(f"{OUT}/q2_results.json", "w"), indent=1)

mean = np.array([np.mean(rb[m]["ratio"]) for m in MS])
lo = np.array([np.min(rb[m]["ratio"]) for m in MS])
hi = np.array([np.max(rb[m]["ratio"]) for m in MS])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].errorbar(MS, mean, yerr=[mean - lo, hi - mean], fmt="o-", capsize=4)
ax[0].axhline(1, color="r", ls="--", lw=1, label="ratio = 1")
ax[0].axvline(rank, color="gray", ls=":", label=f"m = rank(D) = {rank}")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("m"); ax[0].set_ylabel(r"Loss$(D,y,\tilde a)$ / Loss$(D,y,b)$")
ax[0].set_title("Q2(b): ratio (mean, min/max of 5 runs)"); ax[0].legend()
sk = np.array([np.mean(rb[m]["sk_loss"]) for m in MS])
ax[1].plot(MS, sk, "s-", color="#f58518", label=r"mean $\|SD\tilde a-Sy\|^2$")
ax[1].axhline(Lb, color="r", ls="--", lw=1, label=r"Loss$(D,y,b)$")
ax[1].axvline(rank, color="gray", ls=":")
ax[1].set_xscale("log"); ax[1].set_yscale("symlog", linthresh=1e-3)
ax[1].set_xlabel("m"); ax[1].set_ylabel("sketched loss"); ax[1].set_title("Q2(b): sketched loss"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/q2b_ratio.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
for a_, m in zip(ax, [50, 500, 5000]):
    a_.hist(rd[m]["vals"], bins=20, color="#4c78a8", edgecolor="w")
    a_.axvline(1, color="r", ls="--", lw=1)
    a_.set_title(f"m={m}: mean={rd[m]['mean']:.3f}, sd={rd[m]['sd']:.3f}")
    a_.set_xlabel(r"$\|Sv\|^2/\|v\|^2$")
fig.tight_layout(); fig.savefig(f"{OUT}/q2d_hist.png", dpi=150); plt.close(fig)

tm = np.array([np.mean(rb[m]["time"]) for m in MS])
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(MS, tm, "o-", label="CountSketch pipeline (mean of 5)")
ax.axhline(res["a"]["time_mean"], color="r", ls="--", label="full lstsq on D")
ax.set_xscale("log"); ax.set_xlabel("m"); ax.set_ylabel("time (s)"); ax.set_ylim(bottom=0)
ax.set_title("Q2(c): running time"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/q2c_time.png", dpi=150); plt.close(fig)
print("done", flush=True)
