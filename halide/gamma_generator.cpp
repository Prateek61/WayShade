#include "Halide.h"

namespace {

class GammaGenerator : public Halide::Generator<GammaGenerator> {
public:
    Input<Halide::Buffer<uint8_t, 3>>  input{"input"};
    Input<float>                       gamma{"gamma", 1.0f};
    Output<Halide::Buffer<uint8_t, 3>> output{"output"};

    void generate() {
        Halide::Var x, y, c;

        // out = (in/255)^(1/gamma) * 255.
        Halide::Expr norm = Halide::cast<float>(input(x, y, c)) / 255.0f;
        Halide::Expr corrected = Halide::pow(norm, 1.0f / gamma);
        Halide::Expr clamped = Halide::clamp(corrected, 0.0f, 1.0f);
        output(x, y, c) = Halide::cast<uint8_t>(clamped * 255.0f + 0.5f);

        // Accept interleaved RGB(A) buffers: channels is the unit-stride dim,
        // x stride is unconstrained (it equals the channel count at runtime).
        input.dim(0).set_stride(Halide::Expr());
        input.dim(2).set_stride(1);
        output.dim(0).set_stride(Halide::Expr());
        output.dim(2).set_stride(1);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(GammaGenerator, gamma)
