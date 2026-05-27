"""gamma vs an OpenCV LUT of the same (in/255)**(1/value) curve."""

import pytest

import fxlib
import reference
import samples


@pytest.mark.parametrize("value", [0.5, 1.0, 1.8, 2.2, 3.0])
@pytest.mark.parametrize("img_fn", [samples.gradient_rgb, samples.noise_rgb])
def test_gamma_matches_opencv_lut(value, img_fn):
    img = img_fn()
    got = fxlib.apply(img, [("gamma", (value,))])
    ref = reference.gamma(img, value)
    stats = reference.diff_stats(got, ref)
    assert stats["max"] <= 1, stats


def test_gamma_one_is_identity():
    img = samples.noise_rgb()
    got = fxlib.apply(img, [("gamma", (1.0,))])
    assert reference.diff_stats(got, img)["max"] <= 1
