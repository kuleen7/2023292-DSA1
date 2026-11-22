"""Builds the PDF report from results/*.json and results/*.png.  Run:  python build_report.py
EDIT the MEMBERS line below with names / roll numbers before submitting."""
import json
import os
import numpy as np
import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
                                KeepTogether)
from common import OUT

MEMBERS = "Kuleen Mani Tripathi (Roll no. 2023292) — individual submission"
GITHUB_URL = ""   # <-- set to the repository URL, e.g. "https://github.com/<user>/<repo>", then re-run
PDF = os.path.join(OUT, "DSA1_answers.pdf")

fd = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(fd, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", os.path.join(fd, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", os.path.join(fd, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

BASE = ParagraphStyle("b", fontName="DV", fontSize=9, leading=12.5, spaceAfter=4)
H1 = ParagraphStyle("h1", parent=BASE, fontName="DVB", fontSize=14, leading=18, spaceBefore=8, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=BASE, fontName="DVB", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3,
                    textColor=colors.HexColor("#1f3b63"))
SMALL = ParagraphStyle("s", parent=BASE, fontSize=7.5, leading=10)
TITLE = ParagraphStyle("t", parent=BASE, fontName="DVB", fontSize=17, leading=22, alignment=1, spaceAfter=2)
CENTER = ParagraphStyle("c", parent=BASE, alignment=1)

r1 = json.load(open(f"{OUT}/q1_results.json"))
r2 = json.load(open(f"{OUT}/q2_results.json"))
M1 = [5, 10, 15, 20, 25]
M2 = [50, 100, 200, 500, 1000, 2000, 5000]
P = lambda t, s=BASE: Paragraph(t, s)
fmt = lambda x, p=4: f"{x:.{p}f}"


def tbl(data, widths=None, font=8, header=True, zebra=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTNAME", (0, 0), (-1, -1), "DV"), ("FONTSIZE", (0, 0), (-1, -1), font),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f2")), ("FONTNAME", (0, 0), (-1, 0), "DVB")]
    if zebra:
        for i in range(2, len(data), 2):
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f7fa")))
    t.setStyle(TableStyle(st))
    return t


def img(name, w=15.5 * cm):
    from PIL import Image as PI
    pw, ph = PI.open(f"{OUT}/{name}").size
    return Image(f"{OUT}/{name}", width=w, height=w * ph / pw)


S = []
# ================================================================== title + setup
S += [P("Data Science (CSE558) — Assignment: Johnson–Lindenstrauss / random projection", TITLE),
      P(f"Submitted by: {MEMBERS}", CENTER)]
if GITHUB_URL:
    S += [P(f"GitHub repository: <link href='{GITHUB_URL}' color='blue'>{GITHUB_URL}</link>", CENTER)]
S += [Spacer(1, 6)]

S += [P("0. Data, set-up and reproducibility", H1)]
S += [P(f"<b>Data matrix.</b> KDD Cup 99 (10 % version, <font name='DV'>fetch_kddcup99(subset=None, as_frame=True)</font>). "
        f"The three text features were one-hot encoded (3 + 66 + 11 = 80 columns), the other 38 features were converted to "
        f"float (118 columns in total); the two constant columns <i>num_outbound_cmds</i> and <i>is_host_login</i> were "
        f"dropped; every column was then min–max scaled to [0,1]. "
        f"<b>Result: D has n = {r1['n']:,} rows and d = {r1['d']} columns</b> (stored as float32; it is converted to float64 "
        f"for the regression question). {100 * 0.8031:.1f} % of the rows are attacks (y = 1).")]
S += [P(f"<b>rank(D) = {r2['rank']}</b> (not 116): the three smallest singular values of D are ≈ 1.7·10⁻¹¹, 2.7·10⁻¹¹, "
        f"3.6·10⁻¹¹ whereas the next one is 0.106. Three exact linear dependencies exist: the one-hot blocks of "
        f"<i>protocol_type</i>, <i>service</i> and <i>flag</i> each sum to 1 on every row (two independent dependencies), "
        f"and <i>protocol_type_icmp</i> equals the sum of the ICMP-only service columns (ecr_i, eco_i, urp_i, tim_i, red_i, urh_i).")]
S += [P("<b>Reproducibility.</b> Global seed 2026. Every random matrix is generated with "
        "<font name='DV'>numpy.random.default_rng([2026, tag, m, rep])</font> (no sklearn.random_projection); every KMeans "
        "call uses n_clusters = 15, n_init = 5 and a fixed random_state. Environment: Windows 11, 16 logical cores, 15 GB RAM, "
        "Python 3.10.6, numpy 1.24.4, scikit-learn 1.7.1. Code: <i>common.py</i> (data + CountSketch), <i>q1_kmeans.py</i>, "
        "<i>q2_regression.py</i>, <i>make_figs.py</i> (redraws two figures from the saved json), <i>build_report.py</i> (this PDF); "
        "raw outputs in <i>results/</i> (json + logs).")]
S += [P("<b>Notes on implementation.</b> (1) cost(D, ·) is computed by us in float64 (nearest-centre squared distances). "
        f"It equals {fmt(r1['a']['cost'], 2)} for B, whereas sklearn's float32 <i>inertia_</i> prints {fmt(r1['a']['inertia'], 2)} "
        "(0.016 % lower, a float32 rounding effect: we checked that when KMeans is run on a float64 copy of D, its <i>inertia_</i> and our "
        "cost agree to 12 digits, and that our cost function matches a brute-force loop); we use our float64 value everywhere so that all costs "
        "are comparable. We also verified exact reproducibility: re-running Q1 (m = 10, run 1) and Q2 (m = 500, run 1) from the seeds "
        "reproduced the reported cost/loss to every printed digit (<i>verify.py</i>). "
        "(2) CountSketch never forms S. Row i of D is added with sign σ(i) to row h(i) of SD; we do the scatter-add with "
        "<font name='DV'>np.bincount(h, weights=σ·D[:,j])</font> for every column j, which returns exactly the same matrix as "
        "<font name='DV'>np.add.at(SD, h, σ[:,None]*D)</font> (checked: max abs. difference 0.0) but is ≈15× faster "
        "(0.25 s vs 4.2 s) on this numpy version. (3) Times are wall-clock (<font name='DV'>time.perf_counter</font>) on a "
        "multi-core machine; KMeans and BLAS use all cores.", SMALL)]

# ================================================================== Q1
S += [P("Question 1 — JL for k-means clustering (k = 15)", H1)]
a = r1["a"]
S += [P("(a) k-means on D", H2),
      P(f"<b>cost(D, B) = {fmt(a['cost'], 2)}</b>  &nbsp;&nbsp; <b>wall-clock time = {fmt(a['time'], 2)} s</b> "
        f"(KMeans, n_clusters = 15, n_init = 5, random_state = 2026, on the 494,021 × 116 float32 matrix). "
        f"The partition C<sub>D</sub> of this run is used in part (e); cost<sub>D</sub>(C<sub>D</sub>) = {fmt(r1['cost_D_CD'], 2)}.")]

# (b)+(c) table
rows = [["m", "mean ratio\n(b)", "smallest", "largest", "# runs\nratio < 1", "cost(D,B⁺)/cost(D,B)\nmean  [min – max]  (c)"]]
for m in M1:
    b = r1["b"][str(m)]
    rt = np.array(b["ratio"]); rp = np.array(b["ratio_pinv"])
    rows.append([m, fmt(rt.mean()), fmt(rt.min()), fmt(rt.max()), f"{(rt < 1).sum()} / 5",
                 f"{rp.mean():.1f}  [{rp.min():.1f} – {rp.max():.1f}]"])
S += [P("(b), (c) cost ratios (5 independent draws of A for every m)", H2),
      P("Ratio = cost(D, B̂)/cost(D, B) for the centres B̂ obtained by averaging the original rows of D over the clusters of "
        "k-means on E = DAᵀ; the last column is the ratio for B⁺ = {A⁺a}, the pseudo-inverse images of the centres found in R<sup>m</sup>."),
      tbl(rows, [1.2 * cm, 2.6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 5.6 * cm]), Spacer(1, 4)]
rows = [["m"] + [f"run {i + 1}" for i in range(5)]]
for m in M1:
    rows.append([m] + [fmt(v) for v in r1["b"][str(m)]["ratio"]])
S += [P("All 25 individual ratios cost(D, B̂)/cost(D, B):", SMALL), tbl(rows, [1.2 * cm] + [2.3 * cm] * 5, font=7.5),
      Spacer(1, 4), img("q1b_ratio.png", 11.5 * cm)]
S += [P("<b>How the ratio changes with m.</b> For m = 5 the clusters found in the projected space are clearly worse: "
        "mean ratio 1.122 (range 1.020–1.195, every run above 1). Already at m = 10 the mean falls to 1.013, and for "
        "m ≥ 10 the ratio sits within a few per cent of 1 (m = 20: 0.995, m = 25: 0.990). The spread between runs also tends to shrink "
        "with m: the largest − smallest ratio is 0.175 for m = 5, 0.075 for m = 10, 0.153 for m = 15 (one bad run, 1.116), "
        "0.076 for m = 20 and 0.079 for m = 25. So the quality loss decreases (noisily, with only 5 runs) as m grows, which is what JL "
        "predicts: the distortion of squared distances is ≈ √(2/m) per pair, and cost(·) averages many such distortions, so the "
        "ε in (1+ε) shrinks like 1/√m. It is remarkable that m = 10–25 ≪ d = 116 is enough, because the KDD data are essentially "
        "low-dimensional/clustered (a few huge clusters such as smurf and neptune dominate)."),
      P("<b>Ratios below 1.</b> 8 of the 20 runs with m ≥ 10 have ratio < 1 (smallest 0.944 for m = 20, i.e. 5.6 % <i>cheaper</i> "
        "than B). This does not contradict anything: B is not the optimum. k-means (k-means++ seeding + Lloyd, best of 5 starts) "
        "only finds a local optimum of a non-convex problem, and 494k points with k = 15 have many local optima of "
        "different cost (e.g. whether the small attack clusters get their own centre). JL only guarantees "
        "cost(D, B̂) ≤ (1+ε)·OPT, and OPT ≤ cost(D, B). The projected run is just another, somewhat randomised, "
        "route to a local optimum: the clusters of E re-averaged in R<sup>d</sup> (B̂ is a "
        "centroid step on D itself, so it is a valid Lloyd-style update) can land in a better basin than the one the 5 restarts "
        "on D found. The lesson is that the ratio is a noisy quantity around ≈1 that can fall on either side of 1 once m ≳ 10. "
        "<b>Direct evidence that B is only one local optimum:</b> running the very same KMeans call (same seed, n_init = 5) on a float64 copy of D "
        "(a slightly different k-means++ trajectory) gives cost 50,909.4, which is 6.4 % <i>below</i> our reference 54,365.4 — a better "
        "local optimum exists, so ratios of 0.94–0.99 are entirely plausible. (We kept the float32 run as the reference, as the assignment allows "
        "float32, and the same float32 D is used for all runs.)")]

S += [P("(c) B̂ versus B⁺ — which one to use?", H2),
      img("q1c_ratio.png", 10.5 * cm),
      P("B⁺ is <b>far</b> worse: cost(D, B⁺)/cost(D, B) is 60–80 for every run (mean 77.5 at m = 5 down to 63.1 at m = 25), versus ≈ 1 for B̂. "
        "The reason: A⁺A is the orthogonal projector onto the m-dimensional row space of A, so B⁺ = A⁺a is the projection of the "
        "true cluster mean μ<sub>j</sub> onto this tiny random subspace of R<sup>116</sup>. Distances in the sketch are preserved, "
        "but the map back is not a right inverse (m &lt; d, so AA⁺ = I but A⁺A ≠ I); the projected centre keeps only ≈ m/d of "
        "the mean's squared norm and loses all components outside the subspace (all features have mean &gt; 0, so the means are "
        "far from the origin and shrinking them towards 0 is very costly). The ratio improves slowly as m grows, because the "
        "subspace gets larger (m/d = 4 % … 22 %). <b>We would use B̂</b>: it needs only the cluster <i>labels</i> from the cheap "
        "low-dimensional run plus one pass over D to average the rows, it lives in the original space, and it is the object for "
        "which the JL guarantee holds (the partition is near-optimal for D, and the best centre of a partition is its mean).")]

S += [P("(d) Running time", H2)]
rows = [["m", "time of E = DAᵀ + k-means on E (s), mean of 5", "speed-up vs (a)  27.64 s"]]
for m in M1:
    t = np.mean(r1["b"][str(m)]["time"])
    rows.append([m, fmt(t, 2), f"{a['time'] / t:.1f}×"])
S += [tbl(rows, [1.5 * cm, 9 * cm, 5 * cm]), Spacer(1, 4), img("q1d_time.png", 10.5 * cm),
      P("Time grows roughly linearly with m (2.8 s at m = 5 to 5.1 s at m = 25) because the cost of every Lloyd iteration is "
        "O(n·k·m); even at m = 25 the projected pipeline is ≈ 5.4× faster than k-means on D (27.6 s), at m = 5 it is ≈ 10× faster. "
        "The extra cost of drawing A (≈ 10⁻⁴ s) and of multiplying D·Aᵀ (a 494k × 116 by 116 × m product) is negligible. (Forming B̂ and "
        "evaluating cost(D, B̂) take another second or so; they are not part of the quantity asked for.)")]

S += [P("(e) Distortion of pairwise distances versus distortion of the k-means cost", H2),
      P("10⁴ random pairs (i, l) with D<sub>i</sub> ≠ D<sub>l</sub> (the same pairs for all m; pairs of identical rows were redrawn — "
        "the data contain very many duplicate connections). (i) one draw of A per m; (ii) 20 independent draws of A per m with "
        "the clusters C<sub>D</sub> of part (a) held fixed.")]
rows = [["m", "(i) pairs: mean", "sd", "IQR", "min – max", "(ii) cost ratio: mean", "sd", "IQR", "min – max"]]
for m in M1:
    e = r1["e"][str(m)]
    pq, cq = e["pair_q"], e["cost_q"]
    rows.append([m, fmt(e["pair_mean"], 3), fmt(e["pair_sd"], 3), fmt(pq[3] - pq[1], 3), f"{pq[0]:.2f} – {pq[4]:.2f}",
                 fmt(e["cost_mean"], 3), fmt(e["cost_sd"], 3), fmt(cq[3] - cq[1], 3), f"{cq[0]:.2f} – {cq[4]:.2f}"])
S += [tbl(rows, [1.0 * cm, 2.1 * cm, 1.4 * cm, 1.4 * cm, 2.4 * cm, 2.6 * cm, 1.4 * cm, 1.4 * cm, 2.4 * cm], font=7.5),
      Spacer(1, 4), img("q1e_box.png", 13 * cm)]
S += [P("<b>Which has the smaller spread?</b> The <b>k-means cost ratio (ii)</b> has a much smaller spread than the pairwise "
        "distance ratio (i): e.g. sd 0.131 vs 0.355 at m = 5 and 0.048 vs 0.214 at m = 25; the IQR of (ii) is about equal to that of (i) at m = 5 but 2× narrower at m = 10 and ≈ 5× narrower for m ≥ 15, and (i) has far heavier tails. "
        "Both are centred near 1 (JL is unbiased: E‖A x‖² = ‖x‖²), i.e. the boxes are centred around the red line. The reason is "
        "averaging: a single pair's squared length is multiplied by a ≈ χ²<sub>m</sub>/m variable (sd ≈ √(2/m) ≈ 0.63 at m = 5 for a generic direction), "
        "and nothing averages it out, so individual distances can be distorted by a factor of 3 (outliers up to 3.26 and down to 0.02 at m = 5). "
        "The cost, however, equals Σ<sub>j</sub> (1/2|C<sub>j</sub>|) Σ<sub>i,l∈C<sub>j</sub></sub> ‖X<sub>i</sub> − X<sub>l</sub>‖² — a positively weighted average of an enormous number (~10¹¹) of within-cluster "
        "squared distances, so the individual distortions largely average out (not completely, as they all share the same A, which is why a residual spread remains). "
        "Hence the guarantee 'cost of a fixed clustering is preserved up to 1±ε' needs a much smaller m than 'all pairwise distances preserved'."),
      P("<b>Change with m.</b> Both spreads decrease as m grows, roughly like 1/√m: the sd of (ii) falls from 0.131 (m = 5) to 0.114, "
        "0.084, 0.072 and 0.048 (m = 25), i.e. ≈ 2.7× over a 5× larger m (theory: √5 ≈ 2.2×). For (i) the decrease is "
        "visible but less regular (sd 0.355, 0.300, 0.315, 0.300, 0.214; the range shrinks from 0.02–3.3 at m = 5 to 0.35–1.9 at m = 25), because (i) is "
        "computed from <i>one</i> draw of A: the whole cloud of 10⁴ pair ratios is shifted together by the particular A drawn (e.g. at m = 15 "
        "the median is 1.158, at m = 20 it is 0.877), and the heavy mass of near-parallel difference vectors in this "
        "low-dimensional data makes the pairs correlated, so the 10⁴ pairs behave more like a few dozen independent samples. "
        "The (ii) boxes use 20 independent draws and are centred at ≈ 1 (means 0.974, 1.011, 1.005, 1.024, 1.017).")]

# ================================================================== Q2
S += [PageBreak(), P("Question 2 — JL for linear regression (CountSketch)", H1)]
a2 = r2["a"]
S += [P("(a) Full least-squares solution", H2),
      P(f"y = 1 for attacks (80.31 % of rows). <b>Loss(D, y, b) = ‖Db − y‖² = {fmt(a2['loss'], 3)}</b> using "
        f"<font name='DV'>numpy.linalg.lstsq(D, y, rcond=None)</font> on the float64 matrix (it reports rank {r2['rank']}). "
        f"<b>Time to compute b = {fmt(a2['time_mean'], 2)} s</b> (mean of 3 runs: "
        f"{', '.join(fmt(t, 2) for t in a2['times'])} s). Since rank(D) = 113 &lt; 116, lstsq returns the minimum-norm minimiser "
        "(the loss value is the same for any minimiser).")]

rows = [["m", "Loss(D,y,ã)\nmean [min – max]", "ratio to Loss(D,y,b)\nmean", "smallest", "largest", "sketched loss\n‖SDã − Sy‖² (mean)"]]
for m in M2:
    b = r2["b"][str(m)]
    L = np.array(b["loss"]); rt = np.array(b["ratio"])
    rows.append([m, f"{L.mean():.0f}  [{L.min():.0f} – {L.max():.0f}]", fmt(rt.mean(), 3), fmt(rt.min(), 3), fmt(rt.max(), 3),
                 f"{np.mean(b['sk_loss']):.1f}" if m > r2['rank'] else "≈ 0 (< 0.005)"])
S += [P("(b) Ratio and sketched loss (5 fresh CountSketches per m)", H2), tbl(rows, [1.4 * cm, 4.6 * cm, 3.2 * cm, 2 * cm, 2 * cm, 3.5 * cm]),
      Spacer(1, 4), img("q2b_ratio.png", 16 * cm)]
S += [P("Individual runs (loss ã, ratio, sketched loss) are listed in the Appendix table at the end of this report.", SMALL)]
S += [P("<b>What happens.</b> "
        "<u>Above m = rank(D) = 113</u>, the sketched problem is over-determined, ã is unique, and the ratio decreases smoothly to 1 as m grows: "
        "2.27 (m = 200), 1.29 (500), 1.13 (1000), 1.057 (2000), 1.023 (5000); the five runs agree to within a few % at "
        "larger m (the error bars almost vanish). This follows ≈ 1 + r/(m − r) with r = 113: 2.30, 1.29, 1.13, 1.06, 1.02 — the "
        "classical behaviour of a subspace embedding, in which the extra loss is (r/(m−r))·Loss(b). The sketched loss is below the true "
        "minimum: ‖SDã − Sy‖² ≈ (1 − r/m)·Loss(b) (1391, 2422, 2649, 2697, 2887 for the five m &gt; 113 vs predictions 1283, 2283, 2617, 2783, 2883), because ã is "
        "fitted to the sketch with r = 113 free parameters and so adapts to the sketch noise; it approaches Loss(D, y, b) = 2950 from "
        "below as m grows. "
        "<u>Below m = rank(D)</u> (m = 50, 100) SD has fewer rows than the rank of D, the sketched system is under-determined, so lstsq returns an interpolating "
        "minimum-norm solution with ‖SDã − Sy‖² = 0 to numerical precision (the sketched loss is no longer informative at all), and "
        "this overfitted ã is poor on the full data: the ratio is 4.4 (m = 50) and 13.8 (m = 100). The ratio is <i>larger</i> at m = 100 than at "
        "m = 50: the worst point is just under the threshold m ≈ rank, where SD is nearly square/ill-conditioned and the minimum-norm solution "
        "has a huge norm (the 'interpolation peak' of min-norm least squares); the loss of ã comes back down for m &gt; rank. Thus the sharp "
        "transition happens at m = rank(D): the sketch must have at least rank(D) rows to be useful, and a few times more rows to give ratio ≈ 1.1 or better.")]

S += [P("(c) Running time", H2)]
rows = [["m", "time: draw S, form SD & Sy, solve (s)", "speed-up vs full lstsq (4.07 s)"]]
for m in M2:
    t = np.mean(r2["b"][str(m)]["time"])
    rows.append([m, fmt(t, 3), f"{a2['time_mean'] / t:.1f}×"])
S += [tbl(rows, [1.5 * cm, 7 * cm, 6 * cm]), Spacer(1, 4), img("q2c_time.png", 9.5 * cm),
      P("The whole sketch-and-solve pipeline takes 0.27–0.30 s, i.e. ≈ 14–15× faster than the 4.07 s needed for the full solve in (a), "
        "and is almost independent of m. The time is dominated by the single O(n·d) pass over D that forms SD "
        "(≈ 0.25 s for the 116 scatter-adds over 494k rows), whereas solving the m × 116 least-squares problem costs O(m·d²) — "
        "small (the total time grows only from 0.27 s at m = 50 to 0.30 s at m = 5000). The full solve costs O(n·d²) (≈ 4 s). So the speed-up is ≈ d (the dimension). "
        "Per-m times: see table. The saving grows with n.")]

S += [P("(d) Distribution of ‖Sv‖²/‖v‖² for the residual v = Db − y", H2)]
rows = [["m", "mean", "standard deviation", "theory sd  √((2/m)(1 − ‖v‖₄⁴/‖v‖₂⁴))", "min – max (100 draws)"]]
for m in [50, 500, 5000]:
    d_ = r2["d"][str(m)]
    rows.append([m, fmt(d_["mean"]), fmt(d_["sd"]), fmt(d_["theory_sd"]), f"{min(d_['vals']):.3f} – {max(d_['vals']):.3f}"])
S += [tbl(rows, [1.4 * cm, 2.2 * cm, 3.4 * cm, 6.6 * cm, 3.8 * cm]), Spacer(1, 4), img("q2d_hist.png", 16.5 * cm),
      P("<b>Spread vs m.</b> The ratio is unbiased (mean ≈ 1 for all m: 0.994, 1.007, 0.998) and the spread decreases with m like 1/√m: the sd drops from "
        "0.227 (m = 50) to 0.062 (m = 500) and 0.0207 (m = 5000), a factor ≈ 3.7 and ≈ 3.0 for each 10× increase of m (√10 = 3.16). "
        "It agrees with the exact variance of CountSketch, Var(‖Sv‖²) = 2(‖v‖₂⁴ − ‖v‖₄⁴)/m (last column: 0.200, 0.063, 0.020; the sample sd at "
        "m = 50 is 1.9 standard errors above it, which a 100-draw estimate can produce). "
        "At m = 50 individual sketches can shrink or inflate the residual by 40–95 %: 0.61 to 1.95, and the histogram is visibly "
        "right-skewed; at m = 5000 it is a narrow, symmetric bell (0.96–1.05). "
        "This is why a small sketch can give an unreliable loss estimate and the regression ratio of part (b) is large for small m.")]

S += [P("(e) CountSketch versus dense Gaussian sketch on a 50,000-row subset (m = 500)", H2),
      P("50,000 rows chosen without replacement with seed 2031 (<font name='DV'>default_rng(2031).choice(494021, 50000, replace=False)</font>), "
        f"the same subset for both methods and all 5 repetitions; a fresh sketch is drawn in every repetition. Loss(D′, y′, b′) = "
        f"{fmt(r2['e']['loss_bp'], 3)}.")]
rows = [["sketch", "time: draw S, form SD′, Sy′ (s)\nmean ± sd", "(extra) solve\ntime (s)","Loss(D′,y′,ã)/Loss(D′,y′,b′)\nmean ± sd"]]
for key, name in [("count", "(i) CountSketch"), ("dense", "(ii) dense Gaussian N(0,1/m)")]:
    e = r2["e"][key]
    rows.append([name, f"{np.mean(e['t_form']):.4f} ± {np.std(e['t_form']):.4f}", f"{np.mean(e['t_solve']):.4f}",
                 f"{np.mean(e['ratio']):.4f} ± {np.std(e['ratio']):.4f}"])
tc, td = np.mean(r2["e"]["count"]["t_form"]), np.mean(r2["e"]["dense"]["t_form"])
S += [tbl(rows, [4.2 * cm, 4.8 * cm, 2.6 * cm, 5.8 * cm], font=7.5), Spacer(1, 4),
      P(f"Forming the CountSketch takes {tc:.3f} s versus {td:.3f} s for the dense sketch (≈ {td / tc:.0f}× faster), while the quality "
        "is the same within noise: ratios 1.30 ± 0.04 (CountSketch) versus 1.27 ± 0.04 (dense); both are what theory predicts for m = 500, "
        "d = 116: 1 + r/(m − r) ≈ 1.29. The (tiny, insignificant with 5 repetitions) advantage of the dense sketch is the usual one: it is a better "
        "subspace embedding for the same m, but it costs O(m·n·d) rather than O(n·d)."),
      P("<b>Which would we use on the full D?</b> <b>CountSketch.</b> A dense S for the full data would be 500 × 494,021 = 2.5·10⁸ entries "
        "(≈ 2 GB in float64) which have to be generated and then multiplied with D at O(m·n·d) ≈ 3·10¹⁰ flops; scaling the measured "
        f"{td:.2f} s by n/50,000 = 9.9 gives ≈ {td * 9.88:.1f} s, i.e. no faster than just solving the full problem exactly ({a2['time_mean']:.2f} s), so "
        f"the sketch would bring no benefit — whereas the CountSketch needs no storage of S, only O(n) random numbers, and costs one pass over D "
        f"(measured ≈ 0.28 s for the entire pipeline on the full D at m = 500, part (c)) at essentially the same accuracy. "
        "If a better accuracy is needed, it is cheaper to increase m of the CountSketch (m = 2000 gives ratio 1.06 in 0.29 s) than to use a dense sketch.")]

# ================================================================== references + appendix
S += [P("Sources, discussion and acknowledgements", H1),
      P("Individual submission (no group partner); discussion with other groups: <i>none</i>. Resources consulted: scikit-learn documentation for "
        "<i>fetch_kddcup99</i> and <i>KMeans</i> (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html, "
        "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), numpy documentation for <i>lstsq</i>, <i>pinv</i>, "
        "<i>bincount</i>, <i>add.at</i>; background on CountSketch / sketch-and-solve from the course material. "
        "An AI coding assistant (Claude) was used to help write and debug the code and the write-up; all numbers in this report "
        "were produced by running the submitted code.")]
S += [PageBreak(), P("Appendix — per-run results of Q2(b)", H1)]
rows = [["m", "run", "Loss(D,y,ã)", "ratio", "sketched loss", "time (s)"]]
for m in M2:
    b = r2["b"][str(m)]
    for i in range(5):
        rows.append([m, i + 1, fmt(b["loss"][i], 2), fmt(b["ratio"][i], 4), fmt(b["sk_loss"][i], 2), fmt(b["time"][i], 3)])
S += [tbl(rows, [2 * cm, 1.5 * cm, 3.5 * cm, 2.5 * cm, 3.5 * cm, 2.5 * cm], font=7.5)]
S += [P("Per-run Q1(b)/(c) costs: cost(D, B̂) and cost(D, B⁺)", H2)]
rows = [["m", "run", "cost(D,B̂)", "ratio", "cost(D,B⁺)", "ratio B⁺", "time E+k-means (s)"]]
for m in M1:
    b = r1["b"][str(m)]
    for i in range(5):
        rows.append([m, i + 1, fmt(b["cost"][i], 2), fmt(b["ratio"][i], 4), fmt(b["cost_pinv"][i], 1), fmt(b["ratio_pinv"][i], 2),
                     fmt(b["time"][i], 2)])
S += [tbl(rows, [1.5 * cm, 1.3 * cm, 3 * cm, 2.2 * cm, 3 * cm, 2.2 * cm, 3.2 * cm], font=7.5)]


def footer(c, d):
    c.saveState(); c.setFont("DV", 7); c.drawCentredString(A4[0] / 2, 1 * cm, f"CSE558 Assignment — page {d.page}"); c.restoreState()


doc = SimpleDocTemplate(PDF, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.6 * cm, title="CSE558 Assignment")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", PDF)
