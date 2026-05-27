"""rounded vs a numpy transcription of the Inigo Quilez rounded-box SDF mask.
RGBA only; the RGB channels must pass through untouched and only alpha changes."""

import numpy as np
import pytest

import fxlib
import reference
import samples


@pytest.mark.parametrize("radius,softness", [(8.0, 1.0), (16.0, 1.0), (24.0, 2.0), (16.0, 0.5)])
def test_rounded_matches_reference(radius, softness):
    img = samples.sprite_rgba()
    got = fxlib.apply(img, [("rounded", (radius, softness))])
    ref = reference.rounded(img, radius, softness)
    assert reference.diff_stats(got, ref)["max"] <= 1


def test_rounded_only_touches_alpha():
    img = samples.sprite_rgba()
    got = fxlib.apply(img, [("rounded", (16.0, 1.0))])
    assert np.array_equal(got[..., :3], img[..., :3])


def test_rounded_clears_corners_keeps_center():
    img = samples.sprite_rgba()
    got = fxlib.apply(img, [("rounded", (20.0, 1.0))])
    h, w = img.shape[:2]
    assert got[0, 0, 3] == 0                 # corner fully outside the rounded box
    assert got[h // 2, w // 2, 3] == 255     # center unchanged
