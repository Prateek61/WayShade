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
