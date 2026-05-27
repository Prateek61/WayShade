"""shadow vs a numpy transcription of the blur-alpha / offset / tint / source-over
composite. The Gaussian on alpha uses the same 65-tap kernel as the generator but
with a zero (constant_exterior) boundary rather than edge-replicate. Tolerance is
+/-2: the final divide by out_a mildly amplifies float rounding."""

import numpy as np
import pytest

import fxlib
import reference
import samples


@pytest.mark.parametrize("sigma,dx,dy,tint,opacity", [
    (6.0, 0, 8, (0, 0, 0), 0.5),       # classic soft black drop shadow
    (10.0, 6, 6, (0, 0, 0), 0.7),      # diagonal offset
    (8.0, 0, 0, (40, 0, 120), 0.6),    # tinted, no offset
    (4.0, -5, 4, (0, 0, 0), 1.0),      # negative dx, full opacity
])
def test_shadow_matches_reference(sigma, dx, dy, tint, opacity):
    img = samples.sprite_rgba()
    got = fxlib.apply(img, [("shadow", (sigma, dx, dy, tint[0], tint[1], tint[2], opacity))])
    ref = reference.shadow(img, sigma, dx, dy, tint, opacity)
    assert reference.diff_stats(got, ref)["max"] <= 2


def test_shadow_appears_in_transparent_region():
    img = samples.sprite_rgba()
    got = fxlib.apply(img, [("shadow", (8.0, 0, 10, 0, 0, 0, 0.8))])
    # Just below the opaque rectangle the field was transparent; the shadow should
    # have raised alpha there.
    h, w = img.shape[:2]
    band = got[h - 16:h - 4, w // 4:3 * w // 4, 3]
    assert band.max() > 0


def test_shadow_noop_on_fully_opaque_image():
    # An RGB image promoted to RGBA is opaque everywhere, so source-over hides the
    # shadow entirely. See CLAUDE.md: this is by design.
    rgb = samples.gradient_rgb()
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    got = fxlib.apply(rgba, [("shadow", (8.0, 0, 8, 0, 0, 0, 0.5))])
    assert reference.diff_stats(got, rgba)["max"] <= 1
