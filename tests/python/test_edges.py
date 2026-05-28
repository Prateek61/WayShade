"""Edge-case coverage: degenerate and odd-shaped images, the large frame, and the
libfx alpha contract. The blurs read far outside these tiny bounds, so they pin the
TailStrategy::GuardWithIf fixes (1x1 / odd images used to crash, or return all-zeros
on GPU) and confirm channel count survives end to end through the C ABI."""

import numpy as np
import pytest

import fxlib
import reference
import samples

# 1x1, single column/row, tiny (both dims under the 65-tap blur reach), non-power-of-2,
# and a large HD frame. The first four stress boundary/tail handling; the last proves
# nothing chokes at real resolution.
DIMS = [(1, 1), (1, 7), (7, 1), (7, 5), (97, 61), (1920, 1080)]

# RGB effects: (name, args, reference-fn or None, max-channel tolerance).
RGB_CASES = [
    ("gamma",    (2.2,),          lambda im: reference.gamma(im, 2.2),           1),
    ("gaussian", (4.0,),          lambda im: reference.gaussian(im, 4.0),        2),
    ("color",    (1.1, 1.3, 0.7), lambda im: reference.color(im, 1.1, 1.3, 0.7), 1),
    ("kawase",   (1.0,),          None,                                          None),
]

# Alpha effects. shadow's tolerance is looser: its final divide by out_a amplifies
# rounding, more so where a whole tiny image is "edge".
RGBA_CASES = [
    ("rounded", (8.0, 1.0),                lambda im: reference.rounded(im, 8.0, 1.0),                 1),
    ("shadow",  (6.0, 0, 4, 0, 0, 0, 0.6), lambda im: reference.shadow(im, 6.0, 0, 4, (0, 0, 0), 0.6), 3),
]


@pytest.mark.parametrize("w,h", DIMS)
@pytest.mark.parametrize("name,args,ref_fn,tol", RGB_CASES)
def test_rgb_effect_edge_dims(name, args, ref_fn, tol, w, h):
    img = samples.noise_rgb(w, h)  # noise is valid at any dim (no 1/(w-1) ramp)
    got = fxlib.apply(img, [(name, args)])
    assert got.shape == img.shape and got.dtype == np.uint8
    assert got.any()  # a correct transform of non-zero noise is never all-zeros
    if ref_fn is not None:
        stats = reference.diff_stats(got, ref_fn(img))
        assert stats["max"] <= tol, (name, w, h, stats)


@pytest.mark.parametrize("w,h", DIMS)
@pytest.mark.parametrize("name,args,ref_fn,tol", RGBA_CASES)
def test_alpha_effect_edge_dims(name, args, ref_fn, tol, w, h):
    img = samples.rgba_any(w, h)
    got = fxlib.apply(img, [(name, args)])
    assert got.shape == img.shape and got.dtype == np.uint8
    stats = reference.diff_stats(got, ref_fn(img))
    assert stats["max"] <= tol, (name, w, h, stats)


@pytest.mark.parametrize("name,args", [
    ("rounded", (16.0, 1.0)),
    ("shadow",  (6.0, 0, 4, 0, 0, 0, 0.6)),
])
def test_alpha_effect_rejects_rgb(name, args):
    # libfx requires 4-channel input for alpha effects and must NOT silently promote
    # RGB->RGBA the way the CLI does (CLAUDE.md: that's a CLI affordance, not ABI).
    rgb = samples.noise_rgb(32, 24)
    with pytest.raises(fxlib.FxError):
        fxlib.apply(rgb, [(name, args)])


def test_alpha_effects_on_fully_transparent():
    # Alpha 0 everywhere: rounded can only scale 0, and shadow has no source or
    # shadow alpha to composite, so both leave the image fully transparent.
    img = np.zeros((40, 40, 4), np.uint8)
    img[..., :3] = 128
    for name, args in [("rounded", (12.0, 1.0)), ("shadow", (6.0, 0, 4, 0, 0, 0, 0.6))]:
        got = fxlib.apply(img, [(name, args)])
        assert got.shape == img.shape
        assert int(got[..., 3].max()) == 0, name
