"""Tests for features.py: one test group per formula."""

import cv2
import numpy as np
import pytest

from features import (
    chroma_std,
    colorfulness,
    contrast,
    glare_blob_count,
    glare_largest_blob_area_ratio,
    high_frequency_energy_ratio,
    long_line_ratio,
    noise_estimate,
    patch_aggregate,
    orientation_entropy,
    perpendicular_energy_ratio,
    prewitt_energy,
    scharr_energy,
    sharpness,
    sobel_energy,
    specular_highlight_ratio,
)


def test_sharpness_uniform_image_is_exactly_zero() -> None:
    # The Laplacian kernel sums to zero, so convolving a constant image
    # with it gives exactly zero at every pixel regardless of border
    # handling -- this is an exact, hand-derivable result, not just
    # "should be small".
    flat = np.full((20, 20), 128, dtype=np.uint8)
    assert sharpness(flat) == pytest.approx(0.0, abs=1e-9)


def test_sharpness_is_higher_for_a_sharp_edge_than_a_blurred_one() -> None:
    # A hard step-edge has more high-frequency content than the same edge
    # after Gaussian blurring -- blurring is defined by suppressing exactly
    # that content, so the sharp version must score higher.
    sharp = np.zeros((64, 64), dtype=np.uint8)
    sharp[:, 32:] = 255
    blurred = cv2.GaussianBlur(sharp, ksize=(15, 15), sigmaX=5.0)

    assert sharpness(sharp) > sharpness(blurred)


def test_sharpness_rejects_non_grayscale_input() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        sharpness(rgb)


# --- edge operators ---------------------------------------------------------
# Parametrized because all three share identical behaviour by construction:
# each kernel pair sums to zero, so a constant image gives an exact zero, and
# blurring must reduce gradient magnitude. Kept as one block rather than nine
# near-identical copies.

@pytest.mark.parametrize("edge_energy", [sobel_energy, scharr_energy, prewitt_energy])
def test_edge_energy_uniform_image_is_exactly_zero(edge_energy) -> None:
    flat = np.full((20, 20), 128, dtype=np.uint8)
    assert edge_energy(flat) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("edge_energy", [sobel_energy, scharr_energy, prewitt_energy])
def test_edge_energy_is_higher_for_a_sharp_edge_than_a_blurred_one(edge_energy) -> None:
    sharp = np.zeros((64, 64), dtype=np.uint8)
    sharp[:, 32:] = 255
    blurred = cv2.GaussianBlur(sharp, ksize=(15, 15), sigmaX=5.0)

    assert edge_energy(sharp) > edge_energy(blurred)


@pytest.mark.parametrize("edge_energy", [sobel_energy, scharr_energy, prewitt_energy])
def test_edge_energy_rejects_non_grayscale_input(edge_energy) -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        edge_energy(rgb)


# --- patch_aggregate ----------------------------------------------------------

def test_patch_aggregate_std_is_exactly_zero_for_a_uniform_image() -> None:
    # Every patch is identical, so the spread across patches is exactly zero
    # regardless of which formula is applied -- an exact, hand-derivable result.
    flat = np.full((256, 256), 100, dtype=np.uint8)
    assert patch_aggregate(flat, contrast, grid=8, agg="std") == pytest.approx(0.0, abs=1e-12)


def test_patch_aggregate_max_exceeds_mean_when_one_patch_differs() -> None:
    # A single bright square confined to one patch is the whole point of the max
    # aggregation: it registers the worst region rather than diluting it across
    # the image the way a global statistic would.
    # The square must be SMALLER than a patch (32x32 here), not equal to it:
    # a bright square exactly filling a patch leaves that patch uniform, so its
    # contrast is zero like everywhere else and max == mean.
    img = np.zeros((256, 256), dtype=np.uint8)
    img[0:16, 0:16] = 255  # half of the first 32x32 patch

    hottest = patch_aggregate(img, contrast, grid=8, agg="max")
    average = patch_aggregate(img, contrast, grid=8, agg="mean")
    assert hottest > average


def test_patch_aggregate_mean_approximates_the_global_feature() -> None:
    # The control: averaging patches should land near the global value, which is
    # why `mean` carries no new information and `std`/`max` are the useful ones.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)

    assert patch_aggregate(img, contrast, grid=8, agg="mean") == pytest.approx(
        contrast(img), rel=0.1
    )


def test_patch_aggregate_works_on_rgb_input() -> None:
    # Slicing leaves the channel axis intact, so colour formulas patch too.
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    assert patch_aggregate(img, chroma_std, grid=8, agg="std") == pytest.approx(0.0, abs=1e-9)


def test_patch_aggregate_rejects_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match="unknown aggregation"):
        patch_aggregate(np.zeros((256, 256), dtype=np.uint8), contrast, agg="median")


def test_patch_aggregate_rejects_a_grid_that_makes_unmeasurable_patches() -> None:
    with pytest.raises(ValueError, match="too small"):
        patch_aggregate(np.zeros((16, 16), dtype=np.uint8), contrast, grid=8)


# --- input guards -------------------------------------------------------------
# These protect against a silent wrong answer rather than a crash: cv2.cvtColor
# puts HSV V/S in [0, 1] for float input, so a float image would return an
# all-False glare mask and score 0.0 without complaint.

@pytest.mark.parametrize(
    "fn, shape",
    [(specular_highlight_ratio, (10, 10, 3)),
     (glare_blob_count, (10, 10, 3)),
     (glare_largest_blob_area_ratio, (10, 10, 3)),
     (chroma_std, (10, 10, 3)),
     (long_line_ratio, (10, 10))],
)
def test_uint8_dependent_features_reject_float_input(fn, shape) -> None:
    with pytest.raises(ValueError, match="uint8"):
        fn(np.zeros(shape, dtype=np.float64))


def test_noise_estimate_rejects_images_too_small_to_measure() -> None:
    # the (H-2)(W-2) denominator collapses to zero at 2px
    with pytest.raises(ValueError, match="too small"):
        noise_estimate(np.zeros((2, 50), dtype=np.uint8))


def _vertical_edge(size: int = 64) -> np.ndarray:
    """Left half black, right half white: every gradient points along x."""
    img = np.zeros((size, size), dtype=np.uint8)
    img[:, size // 2:] = 255
    return img


# --- orientation_entropy ------------------------------------------------------

def test_orientation_entropy_uniform_image_is_zero() -> None:
    # No gradients anywhere, so orientation is undefined; documented as 0.0.
    assert orientation_entropy(np.full((32, 32), 128, dtype=np.uint8)) == pytest.approx(0.0)


def test_orientation_entropy_is_exactly_zero_for_a_single_straight_edge() -> None:
    # The image is constant along y, so gy is exactly zero everywhere and every
    # gradient lands in the same orientation bin. One bin holds all the
    # probability mass, and -1*log(1) = 0, so this is exact, not approximate.
    assert orientation_entropy(_vertical_edge()) == pytest.approx(0.0, abs=1e-9)


def test_orientation_entropy_is_near_maximal_for_random_noise() -> None:
    # Noise has no preferred direction, so energy spreads evenly across all 36
    # bins -- entropy of a uniform distribution, which normalises to 1.0.
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    assert orientation_entropy(noise) > 0.95


def test_orientation_entropy_rejects_non_grayscale_input() -> None:
    with pytest.raises(ValueError):
        orientation_entropy(np.zeros((10, 10, 3), dtype=np.uint8))


# --- perpendicular_energy_ratio -----------------------------------------------

def test_perpendicular_energy_ratio_is_one_for_a_single_straight_edge() -> None:
    # All energy sits in the dominant bin (and none in its perpendicular
    # partner), so dominant + perpendicular accounts for the entire total.
    assert perpendicular_energy_ratio(_vertical_edge()) == pytest.approx(1.0, abs=1e-9)


def test_perpendicular_energy_ratio_is_high_for_a_rectangle() -> None:
    # A rectangle outline puts its energy at exactly two perpendicular
    # orientations -- the cue this feature exists to detect.
    img = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(img, (24, 24), (104, 104), color=255, thickness=2)
    assert perpendicular_energy_ratio(img) > 0.8


def test_perpendicular_energy_ratio_is_low_for_random_noise() -> None:
    # Energy spread evenly over 36 bins means dominant + perpendicular should
    # capture only ~2/36 of the total.
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    assert perpendicular_energy_ratio(noise) < 0.2


def test_perpendicular_energy_ratio_separates_a_frame_from_noise() -> None:
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    framed = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(framed, (24, 24), (104, 104), color=255, thickness=2)

    assert perpendicular_energy_ratio(framed) > perpendicular_energy_ratio(noise)


def test_perpendicular_energy_ratio_rejects_non_grayscale_input() -> None:
    with pytest.raises(ValueError):
        perpendicular_energy_ratio(np.zeros((10, 10, 3), dtype=np.uint8))


def test_scharr_energy_exceeds_sobel_on_the_same_edge() -> None:
    # Scharr's kernel coefficients (+-3, +-10) are larger than Sobel's
    # (+-1, +-2), so on identical input its raw gradient magnitudes -- and
    # therefore their variance -- must come out larger. Guards against the
    # two accidentally being wired to the same operator.
    edge = np.zeros((64, 64), dtype=np.uint8)
    edge[:, 32:] = 255

    assert scharr_energy(edge) > sobel_energy(edge)



# --- noise_estimate ---------------------------------------------------------

def test_noise_estimate_uniform_image_is_exactly_zero() -> None:
    # The Immerkaer kernel [[1,-2,1],[-2,4,-2],[1,-2,1]] sums to zero, so a
    # constant image convolves to exactly zero everywhere, regardless of
    # border handling (the interior crop sidesteps border effects anyway).
    flat = np.full((20, 20), 100, dtype=np.uint8)
    assert noise_estimate(flat) == pytest.approx(0.0, abs=1e-9)


def test_noise_estimate_rises_with_added_noise() -> None:
    rng = np.random.default_rng(0)
    smooth = np.full((64, 64), 128, dtype=np.float64)
    noisy = np.clip(smooth + rng.normal(0, 25, size=smooth.shape), 0, 255)

    assert noise_estimate(noisy.astype(np.uint8)) > noise_estimate(smooth.astype(np.uint8))


# --- contrast ----------------------------------------------------------------

def test_contrast_matches_hand_computed_std() -> None:
    # values [0, 0, 255, 255]: mean=127.5, population std = 127.5 exactly.
    img = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    assert contrast(img) == pytest.approx(127.5)


def test_contrast_uniform_image_is_zero() -> None:
    assert contrast(np.full((10, 10), 200, dtype=np.uint8)) == pytest.approx(0.0)


# --- colorfulness --------------------------------------------------------------

def test_colorfulness_grayscale_image_is_exactly_zero() -> None:
    # R == G == B everywhere means both opponent-colour axes are exactly
    # zero at every pixel, so both terms of the formula vanish.
    gray_rgb = np.full((10, 10, 3), 150, dtype=np.uint8)
    assert colorfulness(gray_rgb) == pytest.approx(0.0, abs=1e-9)


def test_colorfulness_matches_hand_computed_value() -> None:
    # 1x2 image: pixel1=(100,100,100) [gray], pixel2=(200,100,50) [colourful]
    # rg = R-G = [0, 100] -> mean=50, std=50
    # yb = 0.5*(R+G)-B = [0, 100] -> mean=50, std=50
    # colorfulness = sqrt(50^2+50^2) + 0.3*sqrt(50^2+50^2) = 1.3*sqrt(5000)
    img = np.array([[[100, 100, 100], [200, 100, 50]]], dtype=np.uint8)
    expected = 1.3 * np.sqrt(5000)
    assert colorfulness(img) == pytest.approx(expected, rel=1e-6)


# --- specular_highlight_ratio ------------------------------------------------

def test_specular_highlight_ratio_matches_known_fraction() -> None:
    # Pure white (255,255,255) -> HSV S=0, V=255: passes (V>240, S<30).
    # (100,150,200) -> V=200, S=127 (delta/max*255): fails on S alone.
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    flat = img.reshape(-1, 3)
    flat[:4] = [255, 255, 255]
    flat[4:] = [100, 150, 200]
    img = flat.reshape(3, 3, 3)

    assert specular_highlight_ratio(img) == pytest.approx(4 / 9)


def test_specular_highlight_ratio_zero_when_no_glare() -> None:
    img = np.full((10, 10, 3), (100, 150, 200), dtype=np.uint8)
    assert specular_highlight_ratio(img) == pytest.approx(0.0)


# --- glare blob features ------------------------------------------------------

def _image_with_two_squares(size: int, small_side: int, large_side: int) -> np.ndarray:
    """A coloured background with two separated pure-white squares."""
    img = np.full((size, size, 3), (100, 150, 200), dtype=np.uint8)
    img[0:small_side, 0:small_side] = (255, 255, 255)
    img[size - large_side:, size - large_side:] = (255, 255, 255)
    return img


def test_glare_blob_count_counts_separated_bright_regions() -> None:
    img = _image_with_two_squares(size=20, small_side=2, large_side=4)
    assert glare_blob_count(img) == pytest.approx(2.0)


def test_glare_blob_count_zero_when_no_glare() -> None:
    img = np.full((10, 10, 3), (100, 150, 200), dtype=np.uint8)
    assert glare_blob_count(img) == pytest.approx(0.0)


def test_glare_largest_blob_area_ratio_picks_the_bigger_square() -> None:
    size, large_side = 20, 5
    img = _image_with_two_squares(size=size, small_side=2, large_side=large_side)
    expected = (large_side * large_side) / (size * size)
    assert glare_largest_blob_area_ratio(img) == pytest.approx(expected)


def test_glare_largest_blob_area_ratio_zero_when_no_glare() -> None:
    img = np.full((10, 10, 3), (100, 150, 200), dtype=np.uint8)
    assert glare_largest_blob_area_ratio(img) == pytest.approx(0.0)


# --- long_line_ratio -----------------------------------------------------------

def test_long_line_ratio_zero_for_a_blank_image() -> None:
    # No gradient anywhere -> Canny finds no edges -> no lines detected.
    blank = np.full((100, 100), 128, dtype=np.uint8)
    assert long_line_ratio(blank) == pytest.approx(0.0)


def test_long_line_ratio_high_for_an_image_with_a_long_drawn_line() -> None:
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(img, (10, 10), (90, 90), color=255, thickness=1)
    assert long_line_ratio(img) > 0.5


# --- high_frequency_energy_ratio ------------------------------------------------

def test_high_frequency_energy_ratio_uniform_image_is_exactly_zero() -> None:
    # FFT of a constant signal is a single delta at zero frequency (the
    # exact centre after fftshift), which lies inside any positive-radius
    # low-frequency disk -- so energy outside it is exactly zero.
    flat = np.full((32, 32), 100, dtype=np.uint8)
    assert high_frequency_energy_ratio(flat) == pytest.approx(0.0, abs=1e-9)


def test_high_frequency_energy_ratio_checkerboard_splits_energy_evenly() -> None:
    # A period-2 checkerboard is exactly two frequency components: a DC
    # term (from its mean, 127.5) and a pure Nyquist-frequency term (from
    # its +-127.5 alternation) of *equal* magnitude, with a period that
    # evenly divides the image size so there's no spectral leakage into
    # other bins. So energy splits exactly 50/50 between them -- and the
    # DC sits inside the low-frequency disk while the Nyquist corner sits
    # far outside it, giving a hand-derivable ratio of ~0.5, not "close to
    # 1" as a first guess might assume.
    size = 32
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    checkerboard = (((yy + xx) % 2) * 255).astype(np.uint8)

    assert high_frequency_energy_ratio(checkerboard) == pytest.approx(0.5, abs=0.02)


def test_high_frequency_energy_ratio_higher_for_checkerboard_than_smooth_cosine() -> None:
    # A single full cosine cycle across the image is genuinely periodic (no
    # boundary discontinuity, unlike a tiled linear ramp, which would
    # introduce its own broadband high-frequency content via the Gibbs
    # phenomenon at the wrap-around jump. This cosine has only
    # two nonzero frequency components (DC and the fundamental), both
    # within one pixel of the spectrum centre, so almost all its energy is
    # "low frequency" by construction -- unlike the checkerboard's 50%.
    size = 32
    x = np.arange(size)
    smooth = 127.5 + 127.5 * np.cos(2 * np.pi * x / size)
    smooth_img = np.tile(smooth, (size, 1)).astype(np.uint8)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    checkerboard = (((yy + xx) % 2) * 255).astype(np.uint8)

    assert high_frequency_energy_ratio(checkerboard) > high_frequency_energy_ratio(smooth_img)
    assert high_frequency_energy_ratio(smooth_img) == pytest.approx(0.0, abs=0.02)


# --- chroma_std -----------------------------------------------------------------

def test_chroma_std_grayscale_image_is_exactly_zero() -> None:
    gray_rgb = np.full((10, 10, 3), 150, dtype=np.uint8)
    assert chroma_std(gray_rgb) == pytest.approx(0.0, abs=1e-6)


def test_chroma_std_higher_for_a_colourful_image_than_grayscale() -> None:
    gray_rgb = np.full((10, 10, 3), 150, dtype=np.uint8)
    colourful = np.zeros((10, 10, 3), dtype=np.uint8)
    colourful[:5] = (200, 50, 50)
    colourful[5:] = (50, 50, 200)

    assert chroma_std(colourful) > chroma_std(gray_rgb)
