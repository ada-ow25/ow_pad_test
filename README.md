# Screen-Attack Presentation Attack Detection

Scores face images for how likely they are to be a photograph of a screen or printout
rather than a genuine capture. Outputs a probability per image.

See `report.docx` for the reasoning and results, and `global_analysis.ipynb, patch_analysis.ipyn, external_validation` for the full investigation.

## Quick start (Docker)

```bash
docker build -t iproov-pad .
docker run --rm -v /path/to/images:/data -v "$(pwd)":/out iproov-pad
```

Writes `scores.csv` to the current directory. To pick a different operating point:

```bash
docker run --rm -v /path/to/images:/data -v "$(pwd)":/out iproov-pad \
    --input /data --output /out/scores.csv --bpcer 0.05
```

## Quick start (local)

```bash
conda env create -f environment.yml
conda activate iproov-oliver-whitehead
python infer.py --input /path/to/images --output scores.csv
```

## Output

```csv
filename,score,is_attack
01012.png,0.006166,0
11130.png,0.999997,1
```

- `score` — probability in [0, 1]; higher is more attack-like.
- `is_attack` — `score` thresholded at the chosen operating point.
- Files that fail to decode get an empty score rather than being dropped, so the row count
  always matches the input folder. The failures are also listed on stderr.

Accepts `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`. Non-image files are
ignored.

## Choosing an operating point

`--bpcer` selects a pre-calibrated threshold. **This choice matters more than anything
else in the system:**

| `--bpcer` | genuine users rejected | attacks accepted (APCER) |
|---|---|---|
| `0.001` | ~0.1% | 0.83 (19/23) |
| `0.01` (default) | ~1% | 0.17 (4/23) |
| `0.05` | ~5% | 0.04 (1/23) |
| `0.1` | ~10% | 0.00 (0/23) |

Measured out-of-fold and **pooled over 20 fold splits**, not taken from a single one — a
single split moves the 1% threshold between 0.812 and 0.921, so calibrating from one would
bake in whichever split happened to be used. Median attacks accepted is quoted above; the
spread across splits is 3–7 of 23 at the 1% point.

The default is 1%: 5% catches nearly every attack but rejects one genuine user in twenty,
which is a lot of friction for identity verification. There is no setting that is good on
both axes — that trade-off is the honest result, and `report_assets/det_curve.png` shows it
in full.

The 0.1% threshold sits in the extreme upper tail of 1000 bona fide scores. Pooling splits
removes fold-assignment noise but cannot add information about that tail, which still rests
on a handful of distinct images. Treat it as indicative only.

## How it works

1. **Resize to 1024×1024** (`INTER_AREA`). Not cosmetic — in the training data every bona
   fide image was 1024² and every attack was larger, so raw images are separable on
   dimensions alone. Several features are directly sensitive to pixel dimensions.
2. **Extract 9 features** (`features.py`) covering image quality, colour, glare, framing,
   fine-detail content and edge orientation. Eight of the nine are global statistics; the
   ninth, `hf_energy_pstd`, is computed over an 8×8 grid of patches and measures *spatial
   variation* in fine detail — attacks are unnaturally uniform across the frame. Six other
   features were built and dropped on evidence (VIF, ablation and L1 selection agreed) —
   see `global_analysis.ipynb` §16 and `patch_analysis.ipynb`.
3. **Score** with a logistic regression over rank-transformed features
   (`artefacts/model.joblib`). Regularisation was tuned with nested CV and did not help —
   the untuned default is used deliberately; see report.

A linear model is used because it *outperformed* random forests and every one-class
alternative once rank-based scaling removed the feature skew — so the simplest option is
also the best-performing one, and its coefficients are directly auditable.

## Repository layout

| path | purpose |
|---|---|
| `infer.py` | inference CLI — folder in, CSV out |
| `features.py` | 15 feature formulas; 9 ship, 6 evaluated and dropped |
| `test_features.py` | unit tests (53), one group per formula |
| `train.py` | refits the model, writes artefacts and the DET curve |
| `artefacts/` | fitted model + thresholds and provenance (`config.json`) |
| `global_analysis.ipynb` | the full investigation of the global features |
| `patch_analysis.ipynb` | patch-feature experiment — 12 built, 1 adopted |
| `external_validation.ipynb` | **scores the model on unseen public attack data** |
| `report.md` | technical report |
| `features_cache.csv` | cached global features, so analysis re-runs are instant |
| `features_patch_cache.csv` | cached patch features (slow to compute: ~64 patches/image) |

## Reproducing

```bash
pytest test_features.py     # 53 unit tests
python train.py             # refits from features_cache.csv, rewrites artefacts/
```

`train.py` reads `features_cache.csv`. To regenerate that from the images, set
`FORCE_RECOMPUTE = True` in §1 of `global_analysis.ipynb` and run the feature cell (~8 minutes for
1023 images).

## Limitations

- **It fails on unseen attacks — measured, not suspected.** APCER 1.00 against 15 attack
  videos from two public datasets (`external_validation.ipynb`). Six of nine features point
  the wrong way on that data, carrying 78% of the coefficient weight; the dominant feature
  `noise` is outright inverted. The geometry cues (`perpendicular_ratio`, `glare_area`,
  `orientation_entropy`) did transfer, but carry only 22% of the weight.
- **All 23 training attacks come from one presentation device**, which is the root cause of
  the above: the effective diversity of the attack class is one instrument, not 23 samples.
- **23 attacks gives wide intervals.** Even a perfect 0/23 has a 95% Clopper-Pearson
  interval of [0.00, 0.15] — consistent with a true APCER of 15%.
- **Trained on bona fide at a single resolution.** Every genuine training image was
  1024×1024. Behaviour on genuine images captured at other resolutions is untested.
- **Single-image only.** No temporal or rPPG cues, so video replay gets no benefit from
  motion analysis.
- **Screen/print attacks only.** No coverage of 3D masks or deepfakes.


