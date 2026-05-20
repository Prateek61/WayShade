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

The Halide generator runs at build time and emits `build/halide/fx_gamma.h` and
`build/halide/libfx_gamma.a`, which the C example links against.

## Run the gamma example

```bash
./build/examples/gamma/gamma <input.png> <output.png> <gamma>
```

Convention: `gamma > 1` brightens midtones, `gamma < 1` darkens, `gamma = 1` is identity.

```bash
# Brighter
./build/examples/gamma/gamma input.png brighter.png 2.2

# Darker
./build/examples/gamma/gamma input.png darker.png 0.45

# Identity: should round-trip within ±1 per channel
./build/examples/gamma/gamma input.png identity.png 1.0
```
