# Reference-parity tests

These tests drive the built `libfx.so` through its public C ABI (via `ctypes`)
and check each effect's output against an independent reference.

- **gamma**, **gaussian** are checked against genuine OpenCV primitives
  (`cv2.LUT`, `cv2.GaussianBlur`) configured to match the generators (same fixed
  65-tap window, edge-replicate boundary).
- **color**, **rounded**, **shadow** have no OpenCV equivalent, so they are
  checked against exact numpy transcriptions of the generators' math (float64).
- **kawase** (dual-Kawase) only approximates a Gaussian, so it gets property
  checks (low-pass behaviour, mean preservation, linear-ramp invariance, and
  staying in the Gaussian family) rather than an exact match.

In practice libfx agrees with the references to within 0-1 per channel; the
assert tolerances keep a little headroom for float32-vs-float64 rounding.

## Running

Build libfx first (`cmake --build build`), then from the repo root:

```bash
.venv/bin/python -m pytest tests/python -q
```

Set `FX_LIB_DIR` to point the loader at a non-default build directory; otherwise
it falls back to `build/src/`. The suite is also registered with CTest as
`fx_reference_parity` whenever the project `.venv` exists, so `ctest` runs it
alongside the C ABI test.

Requires `numpy`, `opencv-python`, and `pytest` in the venv.
