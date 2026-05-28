"""Golden-image regression. For three canonical parameter settings per effect we
run libfx on a seeded deterministic input and compare to a committed reference PNG
(tests/reference/). This is the schedule-lock the parity suite does not give: parity
asks "is the output correct vs OpenCV/numpy", golden asks "did the output change at
all".

Regenerate after an intentional change with:  FX_REGEN=1 .venv/bin/python -m pytest tests/python/test_golden.py
"""

import os

import cv2
import numpy as np
import pytest

import fxlib
import samples

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference")

MAX_TOL = 2      # per-channel ceiling: catches gross regressions, tolerates rounding
MEAN_TOL = 0.05  # almost all pixels must be identical, not just bounded

# (effect, label, args, sample-fn). Three settings each, spanning the useful range.
CASES = [
    ("gamma",    "0.5",  (0.5,),               samples.textured_rgb),
    ("gamma",    "1.8",  (1.8,),               samples.textured_rgb),
    ("gamma",    "2.2",  (2.2,),               samples.textured_rgb),
    ("gaussian", "s2",   (2.0,),               samples.textured_rgb),
    ("gaussian", "s4",   (4.0,),               samples.textured_rgb),
    ("gaussian", "s8",   (8.0,),               samples.textured_rgb),
    ("kawase",   "o1",   (1.0,),               samples.textured_rgb),
    ("kawase",   "o2",   (2.0,),               samples.textured_rgb),
    ("kawase",   "o3",   (3.0,),               samples.textured_rgb),
    ("color",    "bri",  (1.2, 1.0, 1.0),      samples.textured_rgb),
    ("color",    "con",  (1.0, 1.4, 1.0),      samples.textured_rgb),
    ("color",    "combo",(1.1, 1.3, 0.7),      samples.textured_rgb),
    ("rounded",  "r8",   (8.0, 1.0),           samples.sprite_rgba),
    ("rounded",  "r16",  (16.0, 1.0),          samples.sprite_rgba),
    ("rounded",  "r24s2",(24.0, 2.0),          samples.sprite_rgba),
    ("shadow",   "soft", (6.0, 0, 8, 0, 0, 0, 0.5),    samples.sprite_rgba),
    ("shadow",   "diag", (10.0, 6, 6, 0, 0, 0, 0.7),   samples.sprite_rgba),
    ("shadow",   "tint", (8.0, 0, 0, 40, 0, 120, 0.6), samples.sprite_rgba),
]

REGEN = os.environ.get("FX_REGEN") == "1"


def _png_path(effect, label):
    return os.path.join(REF_DIR, "%s_%s.png" % (effect, label))


def _write_png(path, rgb):
    # cv2 is BGR(A); our arrays are RGB(A). PNG is lossless, so this round-trips exact.
    bgr = rgb[..., [2, 1, 0]] if rgb.shape[2] == 3 else rgb[..., [2, 1, 0, 3]]
    cv2.imwrite(path, bgr)


def _read_png(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.ndim == 2:
        bgr = bgr[..., None]
    return bgr[..., [2, 1, 0]] if bgr.shape[2] == 3 else bgr[..., [2, 1, 0, 3]]


@pytest.mark.parametrize("effect,label,args,sample_fn", CASES,
                         ids=["%s-%s" % (c[0], c[1]) for c in CASES])
def test_golden(effect, label, args, sample_fn):
    got = fxlib.apply(sample_fn(), [(effect, args)])
    path = _png_path(effect, label)

    if REGEN:
        os.makedirs(REF_DIR, exist_ok=True)
        _write_png(path, got)
        pytest.skip("regenerated " + os.path.basename(path))

    ref = _read_png(path)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    d = np.abs(got.astype(np.int32) - ref.astype(np.int32))
    assert d.max() <= MAX_TOL and d.mean() <= MEAN_TOL, \
        "%s_%s drifted: max=%d mean=%.4f" % (effect, label, d.max(), d.mean())
