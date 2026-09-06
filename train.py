"""Fit the shipped model and write the artefacts inference needs.

Run once: `python train.py`. Produces
  artefacts/model.joblib      fitted pipeline (quantile transform + logistic regression)
  artefacts/config.json       feature order, thresholds, provenance
  report_assets/det_curve.png DET curve for the report

Model choice is justified in report.md: a linear model on rank-transformed
features beat random forests and every one-class alternative, so the simplest
option is also the best-performing one.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in Docker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from infer import FEATURE_ORDER, PATCH_GRID

CACHE = Path("features_cache.csv")
PATCH_CACHE = Path("features_patch_cache.csv")  # patch-aggregated features
ARTEFACTS = Path("artefacts")
ASSETS = Path("report_assets")

# BPCER operating points to calibrate and report. 5% is the value most of the
# analysis used; 1% and 0.1% are included because 5% means rejecting one genuine
# user in twenty, which is not a deployable amount of friction.
BPCER_TARGETS = (0.001, 0.01, 0.05, 0.10)
SHIP_BPCER = 0.01  # the default operating point written into config.json

# Thresholds are calibrated across many fold splits rather than one. A single
# split is a single draw: the stability analysis showed attacks accepted at 1%
# BPCER ranging 3-7 of 23 depending only on the seed, so a threshold taken from
# seed 0 alone inherits whatever that seed happened to produce. Pooling the
# out-of-fold bona fide scores over N seeds averages the fold-assignment noise
# out of the calibration.
#
# What this does NOT fix: we still only have 1000 bona fide images, so the
# extreme upper percentiles rest on a handful of distinct images no matter how
# many times they are re-scored. The 0.1% threshold stays indicative only.
N_CALIBRATION_SEEDS = 20


def build_model():
    """Quantile transform + logistic regression.

    QuantileTransformer rather than StandardScaler because several features are
    heavily right-skewed (glare_area's std is ~18x its robust spread), and
    dividing by an inflated std compresses most images into a narrow band.
    Rank-based scaling removes that, and it is what made the linear model
    outperform random forests.
    """
    return make_pipeline(
        QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=0),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )


def clopper_pearson(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def out_of_fold_scores(X, y, n_splits=5, seed=0):
    """Every image scored by a model that never trained on it."""
    scores = np.empty(len(y))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(X, y):
        model = build_model()
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return scores


def main() -> None:
    if not CACHE.exists():
        raise SystemExit(f"{CACHE} not found -- run the feature extraction in global_analysis.ipynb first")

    df = pd.read_csv(CACHE)
    # Patch-aggregated features live in a separate cache because they are much
    # slower to compute; merge on path so the row alignment cannot drift.
    if PATCH_CACHE.exists():
        patch_df = pd.read_csv(PATCH_CACHE).drop(columns=["label"], errors="ignore")
        df = df.merge(patch_df, on="path", validate="one_to_one")

    missing = [f for f in FEATURE_ORDER if f not in df.columns]
    if missing:
        raise SystemExit(
            f"cached features are missing {missing}. Patch features come from "
            f"{PATCH_CACHE}; regenerate it with patch_analysis.ipynb."
        )

    X = df[FEATURE_ORDER].to_numpy()
    y = (df["label"] == "screen").to_numpy().astype(int)
    print(f"{(y == 0).sum()} bona fide, {(y == 1).sum()} attacks, {len(FEATURE_ORDER)} features")

    # One out-of-fold pass per seed; each is a complete, honest scoring of every
    # image by a model that never trained on it.
    per_seed = np.vstack([out_of_fold_scores(X, y, seed=s)
                          for s in range(N_CALIBRATION_SEEDS)])
    n_attacks = int((y == 1).sum())

    # Thresholds come from bona fide scores only -- never from attack scores --
    # so APCER measured against them stays an honest held-out number rather than
    # one tuned to the attacks we happen to have. Pooling across seeds means the
    # calibration reflects the typical split rather than one arbitrary one.
    pooled_bona = per_seed[:, y == 0].ravel()
    thresholds = {
        str(t): float(np.percentile(pooled_bona, 100 * (1 - t))) for t in BPCER_TARGETS
    }

    print(f"\noperating points (out-of-fold, pooled over {N_CALIBRATION_SEEDS} seeds):")
    rows = []
    for t in BPCER_TARGETS:
        thr = thresholds[str(t)]
        # APCER per seed at the shared threshold, so the spread is visible
        missed = [(per_seed[s][y == 1] <= thr).sum() for s in range(N_CALIBRATION_SEEDS)]
        median_missed = int(np.median(missed))
        lo, hi = clopper_pearson(median_missed, n_attacks)
        # How much would the threshold itself have moved on a single seed?
        single = [np.percentile(per_seed[s][y == 0], 100 * (1 - t))
                  for s in range(N_CALIBRATION_SEEDS)]
        rows.append({
            "BPCER target": t,
            "APCER (median)": median_missed / n_attacks,
            "missed": f"{median_missed}/{n_attacks}",
            "missed range": f"{min(missed)}-{max(missed)}",
            "95% CI": f"[{lo:.2f}, {hi:.2f}]",
            "threshold": round(thr, 4),
            "1-seed thr range": f"[{min(single):.3f}, {max(single):.3f}]",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    aucs = [roc_auc_score(y, per_seed[s]) for s in range(N_CALIBRATION_SEEDS)]
    print(f"\nAUC: {np.mean(aucs):.4f}  (range [{min(aucs):.4f}, {max(aucs):.4f}])")

    # Pooled scores drive the curve too, for the same reason
    scores = per_seed.mean(axis=0)

    # DET curve for the report, over the pooled scores from every seed rather
    # than one split -- a single-split curve is visibly noisier in the tail.
    ASSETS.mkdir(exist_ok=True)
    flat_bona, flat_attack = per_seed[:, y == 0].ravel(), per_seed[:, y == 1].ravel()
    order = np.unique(per_seed)
    bpcer = np.array([(flat_bona > t).mean() for t in order])
    apcer = np.array([(flat_attack <= t).mean() for t in order])
    keep = (bpcer > 0) & (apcer > 0)  # log axes cannot show zeros

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(bpcer[keep], apcer[keep], lw=2, color="#c0392b")
    for t in (0.01, 0.05):
        thr = thresholds[str(t)]
        ax.plot((flat_bona > thr).mean(), (flat_attack <= thr).mean(),
                "o", ms=8, label=f"BPCER {t:.0%} operating point")
    ax.set(xscale="log", yscale="log",
           xlabel="BPCER  (genuine users rejected)",
           ylabel="APCER  (attacks accepted)",
           title=f"DET curve, out-of-fold over {N_CALIBRATION_SEEDS} seeds\nlogistic regression + quantile transform")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "det_curve.png", dpi=150)
    print(f"wrote {ASSETS / 'det_curve.png'}")

    # Final model is fitted on ALL data -- the out-of-fold scores above are what
    # the performance claims rest on; this is the model that ships.
    model = build_model()
    model.fit(X, y)

    ARTEFACTS.mkdir(exist_ok=True)
    dump(model, ARTEFACTS / "model.joblib")

    config = {
        "feature_order": FEATURE_ORDER,
        "image_size": 1024,
        # inference must patch the image identically or hf_energy_pstd shifts
        "patch_grid": PATCH_GRID,
        "thresholds_by_bpcer": thresholds,
        "default_threshold_bpcer": SHIP_BPCER,
        "default_threshold": thresholds[str(SHIP_BPCER)],
        "trained_on": {"bona_fide": int((y == 0).sum()), "attacks": int((y == 1).sum())},
        # thresholds are pooled over this many fold splits, not taken from one
        "calibration_seeds": N_CALIBRATION_SEEDS,
        # A pickled sklearn estimator is version-sensitive: loading it under a
        # different sklearn can warn, or in the worst case unpickle to something
        # that behaves subtly differently. requirements.txt pins the version, and
        # infer.py warns if the runtime disagrees with what is recorded here.
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "out_of_fold_auc": float(np.mean(aucs)),
        "caveat": (
            "All 23 attacks come from a single presentation device, so these "
            "thresholds are calibrated against one attack instrument only."
        ),
    }
    (ARTEFACTS / "config.json").write_text(json.dumps(config, indent=2))
    print(f"wrote {ARTEFACTS / 'model.joblib'} and {ARTEFACTS / 'config.json'}")


if __name__ == "__main__":
    main()
