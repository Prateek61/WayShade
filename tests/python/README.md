# Python test suite

All three suites drive the built `libfx.so` through its public C ABI (via
`ctypes`) and run only the CPU paths, so they are CI-safe.

## Reference-parity (`test_<effect>.py`)

Checks each effect's output against an independent reference.

- **gamma**, **gaussian** vs genuine OpenCV primitives (`cv2.LUT`,
  `cv2.GaussianBlur`) configured to match the generators (same fixed 65-tap
  window, edge-replicate boundary).
- **color**, **rounded**, **shadow** vs exact numpy transcriptions of the
  generators' math in float64 (no OpenCV primitive covers them).
- **kawase** (dual-Kawase) only approximates a Gaussian, so it gets property
  checks (low-pass behaviour, mean preservation, linear-ramp invariance, and
  staying in the Gaussian family) rather than an exact match.

In practice libfx agrees with the references to within 0-1 per channel; the
asserts keep a little headroom for float32-vs-float64 rounding.

## Edge cases (`test_edges.py`)

Every effect at 1x1, single row/col, tiny (7x5), non-power-of-2 (97x61), and
1920x1080. Asserts no crash, channel count preserved, and reference-parity at
that dim where a reference exists. Pins the `TailStrategy::GuardWithIf` fixes
that the blur schedules need on tiny images. Also pins the libfx alpha contract
(`rounded`/`shadow` on a 3-channel image must return `FX_ERR_UNSUPPORTED`, not
silently promote).

## Golden regression (`test_golden.py`)

For three canonical parameter settings per effect, runs libfx on a seeded
deterministic input and compares to a committed PNG under `tests/reference/`.
Complements parity: parity asks "is it correct vs OpenCV/numpy", golden asks
"did it change at all". Tolerance is `max <= 2`, `mean <= 0.05`, low enough to
catch gross schedule regressions but loose enough to survive cross-machine
float-to-u8 rounding.

Regenerate after an intentional change:

```bash
FX_REGEN=1 .venv/bin/python -m pytest tests/python/test_golden.py -q
git diff --stat tests/reference/   # review what moved before committing
```

## Running

Build libfx first (`cmake --build build`), then from the repo root:

```bash
.venv/bin/python -m pytest tests/python -q
```

Set `FX_LIB_DIR` to point the loader at a non-default build directory; otherwise
it falls back to `build/src/`. All three suites are registered with CTest as
`fx_reference_parity` whenever the project `.venv` exists (a no-op for anyone
without it), so `ctest` runs them alongside the C ABI test.

Requires `numpy`, `opencv-python`, and `pytest` in the venv.
