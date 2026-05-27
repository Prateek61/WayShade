"""color vs a numpy transcription of the generator's brightness/contrast/
saturation math (no single OpenCV primitive covers this exact combination)."""

import pytest

import fxlib
import reference
import samples


@pytest.mark.parametrize("params", [
    (1.0, 1.0, 1.0),   # identity
    (1.2, 1.0, 1.0),   # brightness
    (1.0, 1.4, 1.0),   # contrast
    (1.0, 1.0, 1.6),   # saturation
    (1.0, 1.0, 0.0),   # full desaturation -> luma gray
    (1.1, 1.3, 0.7),   # combined
])
@pytest.mark.parametrize("img_fn", [samples.gradient_rgb, samples.noise_rgb])
def test_color_matches_reference(params, img_fn):
    img = img_fn()
    got = fxlib.apply(img, [("color", params)])
    ref = reference.color(img, *params)
    assert reference.diff_stats(got, ref)["max"] <= 1


def test_color_identity_round_trips():
    img = samples.noise_rgb()
    got = fxlib.apply(img, [("color", (1.0, 1.0, 1.0))])
    assert reference.diff_stats(got, img)["max"] <= 1


def test_saturation_zero_is_gray():
    img = samples.gradient_rgb()
    got = fxlib.apply(img, [("color", (1.0, 1.0, 0.0))])
    # R == G == B per pixel (a neutral gray), within rounding.
    assert int(abs(got[..., 0].astype(int) - got[..., 1].astype(int)).max()) <= 1
    assert int(abs(got[..., 1].astype(int) - got[..., 2].astype(int)).max()) <= 1
