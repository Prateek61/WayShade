"""Reference implementations the libfx output is checked against.
"""

import cv2
import numpy as np

# Generated kernel is always 2*max_radius + 1 taps (max_radius GeneratorParam = 32),
# independent of sigma. The reference must use the same fixed window.
MAX_RADIUS = 32
KSIZE = 2 * MAX_RADIUS + 1


def _to_u8(v):
    """Match Halide cast<uint8_t>(clamp(v, lo, hi) + 0.5): round half up, truncate."""
    return (np.clip(v, 0.0, 255.0) + 0.5).astype(np.uint8)


def gamma(arr, value):
    # out = (in/255) ** (1/value), via an OpenCV LUT (the standard gamma curve).
    lut = _to_u8((np.arange(256, dtype=np.float64) / 255.0) ** (1.0 / value) * 255.0)
    return cv2.LUT(arr, lut)


def gaussian(arr, sigma):
    # OpenCV's own separable Gaussian, same fixed 65-tap window and edge-replicate
    # boundary as the generator. Done in float32 then rounded, so the only delta is
    # kernel float precision.
    src = arr.astype(np.float32)
    blurred = cv2.GaussianBlur(src, (KSIZE, KSIZE), sigmaX=sigma, sigmaY=sigma,
                               borderType=cv2.BORDER_REPLICATE)
    return _to_u8(blurred)


def color(arr, brightness, contrast, saturation):
    # adjust(v) = (v*brightness - 0.5)*contrast + 0.5 on normalized RGB, then a
    # Rec.709-luma saturation lerp. Alpha (if present) passes through untouched.
    v = arr[..., :3].astype(np.float64) / 255.0
    v = (v * brightness - 0.5) * contrast + 0.5
    luma = 0.2126 * v[..., 0] + 0.7152 * v[..., 1] + 0.0722 * v[..., 2]
    luma = luma[..., None]
    v = luma + (v - luma) * saturation
    out = arr.copy()
    out[..., :3] = _to_u8(v * 255.0)
    return out


def rounded(arr, radius, softness):
    # Inigo Quilez rounded-box SDF + smoothstep coverage, applied to alpha.
    h, w = arr.shape[:2]
    hx, hy = w * 0.5, h * 0.5
    xs = (np.arange(w, dtype=np.float64) + 0.5) - hx
    ys = (np.arange(h, dtype=np.float64) + 0.5) - hy
    px, py = np.meshgrid(xs, ys)
    dx = np.abs(px) - hx + radius
    dy = np.abs(py) - hy + radius
    mdx, mdy = np.maximum(dx, 0.0), np.maximum(dy, 0.0)
    sdf = np.sqrt(mdx * mdx + mdy * mdy) + np.minimum(np.maximum(dx, dy), 0.0) - radius
    half = max(softness * 0.5, 1e-3)
    t = np.clip((sdf + half) / (2.0 * half), 0.0, 1.0)
    coverage = 1.0 - t * t * (3.0 - 2.0 * t)
    out = arr.copy()
    out[..., 3] = _to_u8(arr[..., 3].astype(np.float64) * coverage)
    return out


def shadow(arr, sigma, dx, dy, tint, opacity):
    # Blur the alpha (zero-padded, same 65-tap kernel), offset it, tint it, then
    # composite the source "over" the shadow with straight alpha.
    h, w = arr.shape[:2]
    k = cv2.getGaussianKernel(KSIZE, sigma, ktype=cv2.CV_64F)  # column, sum-normalized
    alpha = arr[..., 3].astype(np.float64)
    ox, oy = int(round(dx)), int(round(dy))

    # The generator samples the blurred, zero-padded alpha at (x-ox, y-oy), which
    # for pixels near an edge falls outside the image, where the blur is still
    # nonzero (the tail bleeds past the border). Pad with zeros by the kernel reach
    # plus the offset so those out-of-grid samples are real, not clipped to zero.
    p = MAX_RADIUS + max(abs(ox), abs(oy))
    padded = np.zeros((h + 2 * p, w + 2 * p), np.float64)
    padded[p:p + h, p:p + w] = alpha
    blur = cv2.sepFilter2D(padded, cv2.CV_64F, k, k, borderType=cv2.BORDER_CONSTANT)
    shifted = blur[p - oy:p - oy + h, p - ox:p - ox + w]
    sh_a = np.clip(shifted / 255.0, 0.0, 1.0) * opacity

    src_a = alpha / 255.0
    out_a = src_a + sh_a * (1.0 - src_a)
    out = arr.copy()
    for ci in range(3):
        src_c = arr[..., ci].astype(np.float64) / 255.0
        tint_c = tint[ci] / 255.0
        out_c = (src_c * src_a + tint_c * sh_a * (1.0 - src_a)) / np.maximum(out_a, 1e-6)
        out[..., ci] = _to_u8(out_c * 255.0)
    out[..., 3] = _to_u8(out_a * 255.0)
    return out


def diff_stats(a, b):
    """Per-pixel abs-diff summary for asserting and for failure messages."""
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return {
        "max": int(d.max()),
        "mean": float(d.mean()),
        "p999": float(np.percentile(d, 99.9)),
        "frac_gt1": float((d > 1).mean()),
    }
