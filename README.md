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
libraries (`build/halide/fx_gamma.*`, `fx_gaussian_cpu.*`, `fx_gaussian_gpu.*`),
which the `fx_cli` example links against.

## Run the example CLI

```bash
./build/examples/fx_cli/fx_cli <input.png> <output.png> <effects...> [--gpu]
```

Effects apply left-to-right in the order given, so they compose:

```bash
# Gamma correction. gamma > 1 brightens midtones, < 1 darkens, = 1 is identity.
./build/examples/fx_cli/fx_cli input.png brighter.png --gamma 2.2

# Separable Gaussian blur (sigma in pixels).
./build/examples/fx_cli/fx_cli input.png blurred.png --gaussian --sigma 8

# Blur on the GPU (CUDA), then gamma-correct the result.
./build/examples/fx_cli/fx_cli input.png out.png --gaussian --sigma 8 --gamma 0.8 --gpu
```

`--gpu` runs the Gaussian passes on CUDA; gamma always runs on the CPU. On WSL2
the CUDA driver path (`/usr/lib/wsl/lib`) is baked into the binary's RPATH.
