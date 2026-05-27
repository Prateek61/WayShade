"""kawase (dual-Kawase blur) has no OpenCV equivalent and only approximates a
Gaussian, so these are property checks rather than an exact match: it must behave
as a normalized low-pass filter, and stay in the same family as cv2.GaussianBlur."""

import cv2
import numpy as np
import pytest

import fxlib
import samples


def _interior(a, m=8):
    return a[m:-m, m:-m]


@pytest.mark.parametrize("offset", [1.0, 2.0, 3.0])
def test_kawase_reduces_high_frequencies(offset):
    img = samples.noise_rgb()
    got = fxlib.apply(img, [("kawase", (offset,))])
    # A blur removes variance; stronger offset removes more.
    assert _interior(got).var() < _interior(img).var()


@pytest.mark.parametrize("offset", [1.0, 2.0])
def test_kawase_preserves_mean_brightness(offset):
    img = samples.textured_rgb()
    got = fxlib.apply(img, [("kawase", (offset,))])
    # A normalized (sum-to-1) low-pass filter conserves the DC component.
    assert abs(float(_interior(got).mean()) - float(_interior(img).mean())) < 3.0


def test_kawase_near_invariant_on_linear_gradient():
    # A symmetric normalized blur reproduces a linear ramp (away from the edges),
    # which confirms the taps are weighted/normalized correctly.
    img = samples.gradient_rgb()
    got = fxlib.apply(img, [("kawase", (1.0,))])
    d = np.abs(_interior(got).astype(int) - _interior(img).astype(int))
    assert d.max() <= 4, d.max()


def test_kawase_in_gaussian_family():
    # Not an exact match, but the dual-Kawase output should look broadly like a
    # Gaussian blur of comparable strength, not like something unrelated.
    img = samples.textured_rgb()
    got = fxlib.apply(img, [("kawase", (2.0,))])
    ref = cv2.GaussianBlur(img, (0, 0), sigmaX=4.0, borderType=cv2.BORDER_REPLICATE)
    mad = float(np.abs(_interior(got).astype(int) - _interior(ref).astype(int)).mean())
    assert mad < 12.0, mad
