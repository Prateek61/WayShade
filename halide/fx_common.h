#pragma once

#include "Halide.h"

namespace fx {

// Reconfigure a Halide buffer parameter for interleaved RGB(A): channels
// is the unit-stride dim, x stride is unconstrained (it equals the channel
// count at runtime). This is what stb_image produces.
inline void set_interleaved_layout(Halide::OutputImageParam buf) {
    buf.dim(0).set_stride(Halide::Expr());
    buf.dim(2).set_stride(1);
}

}  // namespace fx
