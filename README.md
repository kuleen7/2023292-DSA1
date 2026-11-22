# CSE558 Assignment — JL / random projection on KDD Cup 99

Python 3.10, numpy 1.24, scikit-learn 1.7, matplotlib, pandas, reportlab. Run in this order (from this folder):

    python q1_kmeans.py        # Question 1 -> results/q1_*.json/png   (~4 min)
    python q2_regression.py    # Question 2 -> results/q2_*.json/png   (~1 min)
    python make_figs.py        # redraws Q1(b)/Q2(b) figures from the json
    python build_report.py     # builds results/DSA1_answers.pdf (edit MEMBERS at the top first)

The dataset is downloaded to ../data on first run (fetch_kddcup99) and D is cached there.
Global seed 2026 (see report). `common.py` holds the data-matrix construction and CountSketch.
