"""Deterministic test images. Seeded, so a failure reproduces exactly."""

import numpy as np

W, H = 96, 64


def gradient_rgb(w=W, h=H):
    """Smooth per-channel ramps. A symmetric normalized blur is near-invariant on
    the linear part, which several tests rely on. max(.,1) keeps it defined for a
    1-pixel-wide/tall image (the edge suite passes those in)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dw, dh = max(w - 1, 1), max(h - 1, 1)
    r = x * 255.0 / dw
    g = y * 255.0 / dh
    b = (x / dw + y / dh) * 0.5 * 255.0
    return np.clip(np.stack([r, g, b], axis=-1) + 0.5, 0, 255).astype(np.uint8)


def textured_rgb(w=W, h=H, seed=1):
    """Gradient base plus mid-amplitude noise, so a blur has high frequencies to
    actually remove."""
    rng = np.random.default_rng(seed)
    base = gradient_rgb(w, h).astype(np.int32)
    noise = rng.integers(-40, 41, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def noise_rgb(w=W, h=H, seed=2):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def sprite_rgba(w=W, h=H, margin=18):
    """An opaque colored rectangle on a transparent field. Alpha is 255 inside the
    rectangle and 0 outside, so alpha effects (rounded, shadow) have something to
    act on (a fully-opaque image would make a drop shadow a no-op)."""
    img = np.zeros((h, w, 4), dtype=np.uint8)
    y0, y1 = margin, h - margin
    x0, x1 = margin, w - margin
    img[y0:y1, x0:x1, 0] = 220
    img[y0:y1, x0:x1, 1] = 120
    img[y0:y1, x0:x1, 2] = 60
    img[y0:y1, x0:x1, 3] = 255
    return img


def rgba_any(w=W, h=H, seed=3):
    """RGBA valid at any dimension >= 1 (sprite_rgba's fixed margin breaks on tiny
    images). Noise RGB plus a left-to-right alpha ramp spanning transparent..opaque,
    so alpha effects have a real gradient to act on at every test size."""
    rgb = noise_rgb(w, h, seed)
    ramp = np.round(np.arange(w) * 255.0 / max(w - 1, 1)).astype(np.uint8)
    alpha = np.broadcast_to(ramp, (h, w)).astype(np.uint8)
    return np.dstack([rgb, alpha])
