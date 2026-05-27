# WayShade

GPU/CPU-accelerated visual effects library for Wayland compositors and clients.

> **Status: work in progress.**

## Prerequisites

- `build-essential`, `cmake` (≥ 3.22), `ninja-build`, `git`, `pkg-config`
- Halide binary distribution from <https://github.com/halide/Halide/releases> (Linux
  x86_64). Extract to `~/opt/halide` (or anywhere) and export:
  ```bash
  export Halide_DIR=$HOME/opt/halide/lib/cmake/Halide
  export LD_LIBRARY_PATH=$HOME/opt/halide/lib:$LD_LIBRARY_PATH
  ```

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The Halide generators run at build time and emit per-effect headers and static
libraries (`build/halide/fx_gamma.*`, `fx_gaussian_cpu.*`, `fx_gaussian_gpu.*`,
`fx_kawase_cpu.*`, `fx_kawase_gpu.*`, `fx_shadow_cpu.*`, `fx_shadow_gpu.*`,
`fx_color.*`, `fx_rounded.*`), which the `fx_cli` example links against.

## Run the example CLI

```bash
./build/examples/fx_cli/fx_cli <input.png> <output.png> <effects...> [--gpu]
```

Effects apply left-to-right in the order given, so they compose. Run
`fx_cli --list` for the available effects and their parameters, `--help` for usage.

```bash
# Gamma correction. gamma > 1 brightens midtones, < 1 darkens, = 1 is identity.
./build/examples/fx_cli/fx_cli input.png brighter.png --gamma 2.2

# Separable Gaussian blur (sigma in pixels).
./build/examples/fx_cli/fx_cli input.png blurred.png --gaussian --sigma 8

# Dual-Kawase blur (offset scales the radius). The fast, compositor-style blur.
./build/examples/fx_cli/fx_cli input.png blurred.png --kawase --offset 3 --gpu

# Drop shadow. Blurs the alpha, offsets/tints it, composites the source over.
# Only visible where the source is transparent (an icon or sprite, not a photo).
./build/examples/fx_cli/fx_cli icon.png shadowed.png --shadow --sigma 8 --dy 8 --opacity 0.5

# Color correction (brightness, contrast, saturation; 1.0 each is identity).
./build/examples/fx_cli/fx_cli input.png graded.png --color --brightness 1.1 --saturation 1.3

# Rounded-corner mask. Carves the alpha with an antialiased rounded rectangle.
./build/examples/fx_cli/fx_cli icon.png rounded.png --rounded --radius 16

# Blur on the GPU (CUDA), then gamma-correct the result.
./build/examples/fx_cli/fx_cli input.png out.png --gaussian --sigma 8 --gamma 0.8 --gpu
```

`--gpu` runs the blur and shadow passes on CUDA; gamma, color, and rounded always
run on the CPU. The shadow only appears in transparent areas of the source, so
apply it to an image with an alpha channel rather than an opaque photo. On WSL2 the
CUDA driver path (`/usr/lib/wsl/lib`) is baked into the binary's RPATH.

### Driving the chain from a config file

Instead of inline flags, the whole chain can come from a TOML file
(`--config <file>`, mutually exclusive with inline effects; a CLI `--gpu`
overrides the file's `backend`):

```toml
backend = "gpu"

[[effect]]
name   = "kawase"
offset = 3.0

[[effect]]
name  = "gamma"
value = 0.85
```

```bash
./build/examples/fx_cli/fx_cli input.png out.png --config chain.toml
```

## Use as a library

The effects are also exposed through a small C ABI (`include/fx/fx.h`), built as
`libfx.so`. Create a context, build a pipeline, and run it into a caller-owned
output image:

```c
#include <fx/fx.h>

fx_context_t* ctx;
fx_context_create(FX_BACKEND_CPU, &ctx);

fx_image_t *in, *out;
fx_image_from_data(ctx, w, h, 3, rgb_bytes, &in);  /* 3 = RGB, 4 = RGBA */
fx_image_create(ctx, w, h, 3, &out);

fx_pipeline_t* p;
fx_pipeline_create(ctx, &p);
fx_pipeline_gaussian(p, 8.0f);   /* effects apply in the order appended */
fx_pipeline_gamma(p, 0.8f);
fx_pipeline_run(p, in, out);     /* writes into out, leaves in unchanged */

const unsigned char* result = fx_image_data(out);
/* ... use result ... */

fx_pipeline_destroy(p);
fx_image_destroy(out);
fx_image_destroy(in);
fx_context_destroy(ctx);
```

Handles are opaque and freed by their matching `_destroy`. Every call returns an
`fx_status_t` (`FX_OK` on success), and `fx_context_last_error(ctx)` gives a
detail string. The alpha effects (`rounded`, `shadow`) need a 4-channel image.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

`fx_capi_test` is a pure-C check of the ABI. `fx_reference_parity` compares each
effect against an OpenCV or numpy reference (needs a local `.venv` with `numpy`,
`opencv-python`, and `pytest`, skipped if absent). See `tests/python/README.md`.
