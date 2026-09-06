"""Score a folder of face images for screen-attack likelihood.

    python infer.py --input <folder> --output scores.csv

Writes a CSV of filename, score, is_attack. Score is a probability in [0, 1];
higher means more attack-like. `is_attack` applies the threshold in
artefacts/config.json (calibrated at 1% BPCER by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sklearn
from joblib import load

from features import (
    chroma_std,
    glare_largest_blob_area_ratio,
    high_frequency_energy_ratio,
    long_line_ratio,
    noise_estimate,
    orientation_entropy,
    patch_aggregate,
    perpendicular_energy_ratio,
    prewitt_energy,
)

# 8x8 grid over a 1024px image gives 128px patches -- large enough to contain
# several periods of a moire pattern, small enough that a bezel dominates the
# patches it crosses. Must match what train.py was fitted with.
PATCH_GRID = 8

# Order matters: the model was fitted on columns in this order, so inference
# must reproduce it exactly. train.py imports this same list.
FEATURE_ORDER = [
    "noise",
    "chroma_std",
    "glare_area",
    "line_ratio",
    "hf_energy",
    "prewitt_energy",
    "orientation_entropy",
    "perpendicular_ratio",
    "hf_energy_pstd",
]

_EXTRACTORS = {
    "noise": lambda rgb, gray: noise_estimate(gray),
    "chroma_std": lambda rgb, gray: chroma_std(rgb),
    "glare_area": lambda rgb, gray: glare_largest_blob_area_ratio(rgb),
    "line_ratio": lambda rgb, gray: long_line_ratio(gray),
    "hf_energy": lambda rgb, gray: high_frequency_energy_ratio(gray),
    "prewitt_energy": lambda rgb, gray: prewitt_energy(gray),
    "orientation_entropy": lambda rgb, gray: orientation_entropy(gray),
    "perpendicular_ratio": lambda rgb, gray: perpendicular_energy_ratio(gray),
    "hf_energy_pstd": lambda rgb, gray: patch_aggregate(
        gray, high_frequency_energy_ratio, grid=PATCH_GRID, agg="std"),
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: Path, size: int):
    """Decode and normalise to `size` x `size`.

    Resolution normalisation is load-bearing, not cosmetic: in the training data
    every bona fide image was 1024x1024 and every attack was larger, so a model
    given raw images could separate the classes on dimensions alone. Several
    features (sharpness, line_ratio, hf_energy) are directly sensitive to pixel
    dimensions, so every image must arrive at the same size.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode as an image")
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def extract(path: Path, size: int) -> dict:
    rgb, gray = load_image(path, size)
    return {name: _EXTRACTORS[name](rgb, gray) for name in FEATURE_ORDER}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="folder of images")
    parser.add_argument("--output", default=Path("scores.csv"), type=Path, help="output CSV")
    parser.add_argument("--artefacts", default=Path("artefacts"), type=Path)
    parser.add_argument("--bpcer", default=None,
                        help="operating point, e.g. 0.01; default from config.json")
    args = parser.parse_args(argv)

    if not args.input.is_dir():
        print(f"error: {args.input} is not a directory", file=sys.stderr)
        return 2

    config = json.loads((args.artefacts / "config.json").read_text())
    model = load(args.artefacts / "model.joblib")
    size = config["image_size"]

    # A pickled sklearn estimator is only guaranteed to behave identically under
    # the version it was written by. requirements.txt pins that version, so this
    # should never fire -- but a silent mismatch would produce wrong scores
    # rather than an error, which is the failure mode worth warning about.
    trained_with = config.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        print(f"warning: model was trained with scikit-learn {trained_with}, "
              f"running {sklearn.__version__}; scores may differ", file=sys.stderr)

    key = str(float(args.bpcer)) if args.bpcer else str(config["default_threshold_bpcer"])
    if key not in config["thresholds_by_bpcer"]:
        available = ", ".join(config["thresholds_by_bpcer"])
        print(f"error: no threshold for BPCER {key}; available: {available}", file=sys.stderr)
        return 2
    threshold = config["thresholds_by_bpcer"][key]

    paths = sorted(p for p in args.input.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 1

    rows, failures = [], []
    for path in paths:
        try:
            rows.append({"filename": path.name, **extract(path, size)})
        except Exception as exc:  # one unreadable file shouldn't lose the whole run
            failures.append((path.name, str(exc)))

    if not rows:
        print("error: no images could be processed", file=sys.stderr)
        return 1

    frame = pd.DataFrame(rows)
    scores = model.predict_proba(frame[FEATURE_ORDER].to_numpy())[:, 1]

    out = pd.DataFrame({
        "filename": frame["filename"],
        "score": scores,
        "is_attack": (scores > threshold).astype(int),
    })
    # Failed files are reported with an empty score rather than dropped, so the
    # output row count always matches the input folder.
    for name, _ in failures:
        out.loc[len(out)] = {"filename": name, "score": np.nan, "is_attack": pd.NA}

    out = out.sort_values("filename")
    out.to_csv(args.output, index=False)

    print(f"scored {len(rows)}/{len(paths)} images -> {args.output}")
    print(f"threshold {threshold:.4f} (BPCER {key}); flagged {int(out['is_attack'].sum())} as attacks")
    for name, err in failures:
        print(f"  failed: {name}: {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
