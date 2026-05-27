"""gaussian vs cv2.GaussianBlur with the same fixed 65-tap window and
edge-replicate boundary. Divergence is only kernel float precision and the
round-half (cv2 banker's) vs round-half-up (Halide) tie rule, bounded at +/-2
with essentially everything within +/-1."""

import pytest

import fxlib
import reference
import samples


@pytest.mark.parametrize("sigma", [2.0, 4.0, 8.0, 12.0])
@pytest.mark.parametrize("img_fn", [samples.textured_rgb, samples.noise_rgb])
def test_gaussian_matches_opencv(sigma, img_fn):
    img = img_fn()
    got = fxlib.apply(img, [("gaussian", (sigma,))])
    ref = reference.gaussian(img, sigma)
    stats = reference.diff_stats(got, ref)
    assert stats["max"] <= 2, stats
    assert stats["p999"] <= 1, stats
    assert stats["mean"] < 0.05, stats


def test_gaussian_actually_blurs():
    img = samples.noise_rgb()
    got = fxlib.apply(img, [("gaussian", (4.0,))])
    # A low-pass filter reduces variance and high-frequency energy.
    assert got.var() < img.var()
