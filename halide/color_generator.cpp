#include "Halide.h"
#include "fx_common.h"

namespace {

// Color correction via convenience scalars on normalized RGB.
class ColorGenerator : public Halide::Generator<ColorGenerator> {
public:
    Input<Halide::Buffer<uint8_t, 3>>  input{"input"};
    Input<float>                       brightness{"brightness", 1.0f};
    Input<float>                       contrast{"contrast", 1.0f};
    Input<float>                       saturation{"saturation", 1.0f};
    Output<Halide::Buffer<uint8_t, 3>> output{"output"};

    void generate() {
        using namespace Halide;
        Var x, y, c;

        auto adjust = [&](Expr v) { return (v * brightness - 0.5f) * contrast + 0.5f; };
        Expr r = adjust(cast<float>(input(x, y, 0)) / 255.0f);
        Expr g = adjust(cast<float>(input(x, y, 1)) / 255.0f);
        Expr b = adjust(cast<float>(input(x, y, 2)) / 255.0f);

        Expr luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        r = luma + (r - luma) * saturation;
        g = luma + (g - luma) * saturation;
        b = luma + (b - luma) * saturation;

        Expr rgb = select(c == 0, r, c == 1, g, b);
        output(x, y, c) =
            select(c < 3, cast<uint8_t>(clamp(rgb, 0.0f, 1.0f) * 255.0f + 0.5f),
                          input(x, y, c));

        fx::set_interleaved_layout(input);
        fx::set_interleaved_layout(output);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ColorGenerator, color)
