"""Per-image feature formulas for screen-attack detection.

Each function takes a decoded image array and returns a single float. Kept
as plain functions over arrays (no file I/O, no classes) so each one is
independently testable against a hand-constructed input with a known
answer.
"""

from __future__ import annotations

import cv2
import numpy as np


def sharpness(image_gray: np.ndarray) -> float:
    """Variance of the Laplacian: a standard focus/blur measure.

    The Laplacian responds to intensity changes (edges); its variance
    across the image is high when edges are crisp and collapses toward
    zero as the image blurs, because blurring is exactly what suppresses
    those high-frequency changes. Recapturing a screen through a second
    camera typically loses sharpness relative to a direct live capture, so
    a low value here is one vote toward "attack".

    Reference: Galbally et al., "Image Quality Assessment for Fake
    Biometric Detection", IEEE Trans. Image Processing, 2014.
    """
    _check_gray(image_gray)
    laplacian = cv2.Laplacian(image_gray, ddepth=cv2.CV_64F)
    return float(laplacian.var())


def _check_gray(image_gray: np.ndarray) -> None:
    if image_gray.ndim != 2:
        raise ValueError(f"expected a 2D grayscale array, got shape {image_gray.shape}")


def _check_rgb(image_rgb: np.ndarray) -> None:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB array, got shape {image_rgb.shape}")


def _check_uint8(image: np.ndarray) -> None:
    """Guard the features whose correctness depends on 8-bit input.

    cv2.cvtColor scales HSV/YCrCb differently for float arrays (V and S land in
    [0, 1] rather than [0, 255]), so a float image would silently return an
    all-False glare mask and a zero score rather than raising -- a wrong answer
    delivered confidently, which is the worst failure mode for inference.
    """
    if image.dtype != np.uint8:
        raise ValueError(
            f"expected an 8-bit image (dtype uint8), got {image.dtype}; "
            "colour-space conversion is scaled differently for other dtypes"
        )


# --- Patch aggregation ------------------------------------------------------
#
# Every feature below is a *global* statistic over the whole image, but a screen
# attack is not globally wrong -- it is locally wrong. The bezel is at the
# boundary, moire concentrates in flat regions, glare is a blob. Applying the
# same formula over a grid of patches and aggregating lets the model express
# "part of this image looks wrong" rather than "the average of this image looks
# slightly off".
#
# The aggregation carries the new information, not the patching: `mean` just
# reproduces the global feature, whereas
#   max -- the single most anomalous patch, which is what a localised artefact
#          looks like numerically
#   std -- spatial heterogeneity. A genuine face varies smoothly; a recaptured
#          screen has a sharp discontinuity at the frame and unnaturally uniform
#          texture inside it. Nothing in the global feature set can say this.


def _iter_patches(image: np.ndarray, grid: int, overlap: float):
    """Yield grid x grid patches. Works for 2D grayscale and HxWx3 alike."""
    height, width = image.shape[:2]
    patch_h, patch_w = height // grid, width // grid
    if patch_h < 3 or patch_w < 3:
        raise ValueError(f"grid={grid} gives {patch_h}x{patch_w} patches, too small to measure")
    step_h = max(1, int(patch_h * (1 - overlap)))
    step_w = max(1, int(patch_w * (1 - overlap)))
    for r in range(0, height - patch_h + 1, step_h):
        for c in range(0, width - patch_w + 1, step_w):
            yield image[r:r + patch_h, c:c + patch_w]


def patch_aggregate(image: np.ndarray, fn, grid: int = 8, agg: str = "std",
                    overlap: float = 0.0) -> float:
    """Apply `fn` to each patch of `image` and aggregate across patches.

    `fn` is any of the per-image formulas in this module. On a 1024px image a
    grid of 8 gives 128px patches -- large enough to contain several periods of
    a moire pattern, small enough that a bezel dominates the patches it crosses.
    """
    values = np.array([fn(patch) for patch in _iter_patches(image, grid, overlap)])
    if values.size == 0:
        raise ValueError("no patches produced")
    if agg == "std":
        return float(values.std())
    if agg == "max":
        return float(values.max())
    if agg == "mean":  # included mainly as the control -- approximates the global feature
        return float(values.mean())
    raise ValueError(f"unknown aggregation {agg!r}; expected 'std', 'max' or 'mean'")


# --- Edge operators ---------------------------------------------------------
#
# Added as separate columns rather than by injecting edges into the image and
# re-measuring everything: injection forces every feature to measure a mixture
# of edge and colour signal, whereas separate columns let a model weight the
# two independently. (The injection experiment is in the notebook -- it lost
# information to uint8 clipping.)


def _gradient_energy(gx: np.ndarray, gy: np.ndarray) -> float:
    magnitude = np.sqrt(gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2)
    return float(magnitude.var())


def sobel_energy(image_gray: np.ndarray) -> float:
    """Variance of Sobel gradient magnitude (3x3, derivative with built-in smoothing)."""
    _check_gray(image_gray)
    gx = cv2.Sobel(image_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gy = cv2.Sobel(image_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    return _gradient_energy(gx, gy)


def scharr_energy(image_gray: np.ndarray) -> float:
    """Variance of Scharr gradient magnitude (3x3, better rotational symmetry than Sobel)."""
    _check_gray(image_gray)
    gx = cv2.Scharr(image_gray, ddepth=cv2.CV_64F, dx=1, dy=0)
    gy = cv2.Scharr(image_gray, ddepth=cv2.CV_64F, dx=0, dy=1)
    return _gradient_energy(gx, gy)


_PREWITT_KX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
_PREWITT_KY = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)


def prewitt_energy(image_gray: np.ndarray) -> float:
    """Variance of Prewitt gradient magnitude (3x3, uniform weighting -- no smoothing, so noisier)."""
    _check_gray(image_gray)
    gray_f = image_gray.astype(np.float64)
    gx = cv2.filter2D(gray_f, ddepth=-1, kernel=_PREWITT_KX)
    gy = cv2.filter2D(gray_f, ddepth=-1, kernel=_PREWITT_KY)
    return _gradient_energy(gx, gy)


# --- Edge orientation -------------------------------------------------------
#
# Everything above measures edge *strength*. These measure edge *direction*,
# which is where the bezel cue actually lives: a screen or sheet of paper
# concentrates long edges at two perpendicular angles, whereas a face spreads
# its edges across all angles.

_ORIENTATION_BINS = 36  # 5 degrees per bin across [0, 180)


def _orientation_histogram(image_gray: np.ndarray, n_bins: int = _ORIENTATION_BINS) -> np.ndarray:
    """Magnitude-weighted histogram of edge orientations over [0, 180)."""
    gx = cv2.Sobel(image_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gy = cv2.Sobel(image_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    # Orientation is modulo 180, not 360: an edge at 10 degrees and one at 190
    # are the same edge with the gradient pointing the opposite way.
    orientation = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    # Weighting by magnitude means strong edges dominate and noise (many weak
    # gradients at random angles) contributes little without needing a threshold.
    hist, _ = np.histogram(orientation, bins=n_bins, range=(0.0, 180.0), weights=magnitude)
    return hist


def orientation_entropy(image_gray: np.ndarray) -> float:
    """Shannon entropy of the edge-orientation histogram, normalised to [0, 1].

    0 means every edge points the same way (a single straight line); 1 means
    edges are spread evenly across all angles. Framed content should sit lower
    than an unframed face.

    Returns 0.0 for an image with no gradients at all, where the quantity is
    genuinely undefined -- can't occur for a real photograph.
    """
    _check_gray(image_gray)
    hist = _orientation_histogram(image_gray)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    nonzero = p[p > 0]  # 0*log(0) is 0 by convention, so empty bins just drop out
    return float(-(nonzero * np.log(nonzero)).sum() / np.log(len(hist)))


def perpendicular_energy_ratio(image_gray: np.ndarray) -> float:
    """Share of edge energy in the dominant orientation *and its perpendicular*.

    Targets rectangular framing specifically rather than mere concentration: a
    bezel puts energy at both theta and theta+90, whereas e.g. a striped
    background is concentrated but not perpendicular.

    Known sensitivity: a frame rotated so its edges straddle a bin boundary
    splits energy across two adjacent bins and scores lower than it should.
    Widening to the neighbouring bins would smooth that out if it proves to matter.
    """
    _check_gray(image_gray)
    hist = _orientation_histogram(image_gray)
    total = hist.sum()
    if total == 0:
        return 0.0
    n_bins = len(hist)
    dominant = int(np.argmax(hist))
    perpendicular = (dominant + n_bins // 2) % n_bins  # half the bins == 90 degrees
    return float((hist[dominant] + hist[perpendicular]) / total)


# --- IQA family (Galbally et al. 2014) --------------------------------------


def noise_estimate(image_gray: np.ndarray) -> float:
    """Fast per-pixel noise sigma via Immerkaer's (1996) Laplacian estimator.

    Convolves with a kernel whose response to any smooth (locally planar)
    signal is zero, so what survives is attributable to noise rather than
    image structure. Screen recapture typically adds sensor/compression
    noise from the second camera on top of whatever the display already
    introduced, so a genuine live capture is expected to score lower.

    Reference: J. Immerkaer, "Fast Noise Variance Estimation", Computer
    Vision and Image Understanding, 1996.
    """
    _check_gray(image_gray)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    filtered = cv2.filter2D(image_gray.astype(np.float64), ddepth=-1, kernel=kernel)
    # Crop to the region unaffected by border padding, matching the
    # (H-2)*(W-2) valid-convolution denominator in the closed-form estimator.
    interior = filtered[1:-1, 1:-1]
    height, width = image_gray.shape
    if height <= 2 or width <= 2:
        # the (H-2)(W-2) denominator collapses to zero -- no interior to measure
        raise ValueError(f"image too small for noise estimation: {image_gray.shape}")
    sigma = np.sqrt(np.pi / 2) * np.sum(np.abs(interior)) / (6 * (width - 2) * (height - 2))
    return float(sigma)


def contrast(image_gray: np.ndarray) -> float:
    """RMS (global) contrast: the standard deviation of pixel intensity."""
    _check_gray(image_gray)
    return float(image_gray.astype(np.float64).std())


def colorfulness(image_rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk's (2003) colorfulness metric.

    Combines the spread and offset of two opponent colour axes (red-green
    and yellow-blue). Screens reproduce a narrower, differently-biased
    gamut than skin/scene reflectance under natural light, so this is
    expected to differ systematically between bona fide and screen capture.

    Reference: D. Hasler, S. Suesstrunk, "Measuring Colorfulness in Natural
    Images", Human Vision and Electronic Imaging VIII, 2003.
    """
    _check_rgb(image_rgb)
    rgb = image_rgb.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def _bright_desaturated_mask(image_rgb: np.ndarray, v_thresh: int = 240, s_thresh: int = 30) -> np.ndarray:
    """Pixels that are near-saturated bright and low-chroma: candidate glare/specular reflection."""
    _check_rgb(image_rgb)
    _check_uint8(image_rgb)  # thresholds below are on the 0-255 HSV scale
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation, value = hsv[..., 1], hsv[..., 2]
    return (value > v_thresh) & (saturation < s_thresh)


def specular_highlight_ratio(image_rgb: np.ndarray) -> float:
    """Fraction of pixels that look like specular reflection / glare.

    A pixel counts if it is both very bright and low-saturation (a strongly
    saturated bright colour, e.g. pure red, is not glare; near-white is).
    Diffuse ambient light on skin rarely produces much of this; a flat
    glass/glossy screen surface reflecting a light source will.
    """
    mask = _bright_desaturated_mask(image_rgb)
    return float(mask.mean())


# --- Spot lighting / localized glare (observed cue) -------------------------


def glare_blob_count(image_rgb: np.ndarray) -> float:
    """Number of distinct connected regions of specular/glare pixels.

    Distinguishes "a few concentrated bright spots" (glare/backlight bloom)
    from "generally bright everywhere" (ambient light or overexposure),
    which `specular_highlight_ratio` alone can't tell apart.
    """
    mask = _bright_desaturated_mask(image_rgb).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask, connectivity=8)
    return float(num_labels - 1)  # exclude the background label


def glare_largest_blob_area_ratio(image_rgb: np.ndarray) -> float:
    """Area of the single largest glare blob, as a fraction of the image.

    A single large concentrated bright region is the more distinctive
    "spot lighting" signature; many tiny specks are more likely sensor
    noise or jewellery/eye highlights than screen backlight bloom.
    """
    mask = _bright_desaturated_mask(image_rgb).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:  # only the background component
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]  # row 0 is the background component
    total_pixels = mask.shape[0] * mask.shape[1]
    return float(areas.max() / total_pixels)


# --- Presentation-surface framing / background leakage (observed cue) -------


def long_line_ratio(image_gray: np.ndarray) -> float:
    """Longest detected straight edge, as a fraction of the image diagonal.

    A device bezel or paper edge enclosing a face produces a long, strong,
    straight line that a direct face capture — where edges are mostly the
    curved, broken contours of a face and its background — does not.
    """
    _check_gray(image_gray)
    _check_uint8(image_gray)  # cv2.Canny requires 8-bit input
    edges = cv2.Canny(image_gray, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5
    )
    if lines is None:
        return 0.0
    # cv2 has returned both (N, 1, 4) and (N, 4) across versions; normalize.
    lines = lines.reshape(-1, 4)
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    lengths = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    diagonal = np.sqrt(image_gray.shape[0] ** 2 + image_gray.shape[1] ** 2)
    return float(lengths.max() / diagonal)


# --- Moire / frequency-domain artifacts (observed cue) ----------------------


def high_frequency_energy_ratio(image_gray: np.ndarray, low_freq_frac: float = 0.1) -> float:
    """Fraction of FFT magnitude spectrum energy outside a low-frequency disk.

    A screen's pixel grid beating against the camera's sensor grid (moire)
    concentrates energy at specific mid/high spatial frequencies that a
    smoothly-shaded live face does not produce. `low_freq_frac` sets the
    radius of the excluded low-frequency disk as a fraction of the smaller
    image dimension's half-width.

    Reference: Patel, Han, Jain, "Live Face Video vs. Spoof Face Video: Use
    of Moire Pattern to Detect Replay Video Attacks", ICB 2015.
    """
    _check_gray(image_gray)
    spectrum = np.fft.fftshift(np.fft.fft2(image_gray.astype(np.float64)))
    magnitude = np.abs(spectrum)

    height, width = image_gray.shape
    cy, cx = height / 2, width / 2
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_freq_radius = low_freq_frac * min(height, width) / 2

    total_energy = magnitude.sum()
    if total_energy == 0:
        return 0.0
    high_freq_energy = magnitude[radius > low_freq_radius].sum()
    return float(high_freq_energy / total_energy)


# --- Color-texture (simplified proxy) ---------------------------------------


def chroma_std(image_rgb: np.ndarray) -> float:
    """Combined standard deviation of the two chrominance channels (Cr, Cb).

    A cheap, simplified stand-in for full LBP colour-texture descriptors
    (Boulkenafet et al. 2015/2016): screens reproduce colour with different
    local chrominance variation than skin under direct light. If this proxy
    doesn't carry separating signal once evaluated (see global_analysis.ipynb), a proper
    LBP-in-YCbCr descriptor is the documented next step, not assumed here.
    """
    _check_rgb(image_rgb)
    _check_uint8(image_rgb)  # YCrCb offset differs for float input
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float64)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    return float(np.sqrt(cr.std() ** 2 + cb.std() ** 2))
