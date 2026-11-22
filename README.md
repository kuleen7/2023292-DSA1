# CSE558 Assignment- JL / random projection on KDD Cup 99

Python 3.10, numpy 1.24, scikit-learn 1.7, matplotlib, pandas. Run in this order from the `code/` folder:

    python q1_kmeans.py
    python q2_regression.py
    python make_figs.py
    python verify.py

The dataset is downloaded to ../data on first run (fetch_kddcup99) and D is cached there. Global seed 2026 (see report). `common.py` holds the data-matrix construction and CountSketch.
