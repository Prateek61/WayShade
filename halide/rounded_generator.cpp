#include "Halide.h"
#include "fx_common.h"

namespace {

// Rounded-rectangle alpha mask.
class RoundedGenerator : public Halide::Generator<RoundedGenerator> {
public:
    Input<Halide::Buffer<uint8_t, 3>>  input{"input"};
    Input<float>                       radius{"radius", 16.0f};
    Input<float>                       softness{"softness", 1.0f};
    Output<Halide::Buffer<uint8_t, 3>> output{"output"};

    void generate() {
        using namespace Halide;
        Var x, y, c;

        Expr hx = cast<float>(input.dim(0).extent()) * 0.5f;
        Expr hy = cast<float>(input.dim(1).extent()) * 0.5f;

        // Pixel center relative to the image center.
        Expr px = (cast<float>(x) + 0.5f) - hx;
        Expr py = (cast<float>(y) + 0.5f) - hy;

        // Inigo Quilez rounded-box SDF: < 0 inside, > 0 outside.
        Expr dx = abs(px) - hx + radius;
        Expr dy = abs(py) - hy + radius;
        Expr mdx = max(dx, 0.0f), mdy = max(dy, 0.0f);
        Expr sdf = sqrt(mdx * mdx + mdy * mdy) + min(max(dx, dy), 0.0f) - radius;

        // smoothstep coverage across a softness-wide band centered on the edge.
        Expr half = max(softness * 0.5f, 1e-3f);
        Expr t = clamp((sdf + half) / (2.0f * half), 0.0f, 1.0f);
        Expr coverage = 1.0f - t * t * (3.0f - 2.0f * t);

        Expr a = cast<float>(input(x, y, 3)) * coverage;
        output(x, y, c) =
            select(c == 3, cast<uint8_t>(clamp(a, 0.0f, 255.0f) + 0.5f), input(x, y, c));

        fx::set_interleaved_layout(input);
        fx::set_interleaved_layout(output);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(RoundedGenerator, rounded)
