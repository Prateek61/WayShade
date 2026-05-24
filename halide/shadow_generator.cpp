#include "Halide.h"
#include "fx_common.h"

namespace {

// Drop shadow: blur the alpha, offset it, tint it, composite the original over.
// The loader promotes inputs to RGBA.
class ShadowGenerator : public Halide::Generator<ShadowGenerator> {
public:
    GeneratorParam<int> max_radius{"max_radius", 32};
    GeneratorParam<int> gpu_block_x{"gpu_block_x", 32};
    GeneratorParam<int> gpu_block_y{"gpu_block_y", 8};

    Input<Halide::Buffer<uint8_t, 3>>  input{"input"};
    Input<float>                       sigma{"sigma", 8.0f};
    Input<float>                       dx{"dx", 0.0f};
    Input<float>                       dy{"dy", 8.0f};
    Input<float>                       tint_r{"tint_r", 0.0f};
    Input<float>                       tint_g{"tint_g", 0.0f};
    Input<float>                       tint_b{"tint_b", 0.0f};
    Input<float>                       opacity{"opacity", 0.5f};
    Output<Halide::Buffer<uint8_t, 3>> output{"output"};

    Halide::Var x{"x"}, y{"y"}, c{"c"};
    Halide::Var xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};
    Halide::Var r{"r"};
    Halide::Func kernel_raw{"kernel_raw"}, ksum{"ksum"}, kernel{"kernel"};
    Halide::Func blur_x{"blur_x"}, blur_y{"blur_y"};

    void generate() {
        using namespace Halide;
        const int R = max_radius;

        Expr w = input.dim(0).extent();
        Expr h = input.dim(1).extent();

        // Alpha as a 2D field, transparent (0) outside the image bounds. The
        // repeat_edge clamps the inner read so the wide blur window never indexes
        // `input` out of bounds
        Func in_clamped = BoundaryConditions::repeat_edge(input);
        Func alpha{"alpha"};
        alpha(x, y) = cast<float>(in_clamped(x, y, 3));
        Func alpha_b = BoundaryConditions::constant_exterior(alpha, 0.0f, {{0, w}, {0, h}});

        // Pre-normalized 1D Gaussian, same as gaussian_generator.
        Expr off = cast<float>(r - R);
        Expr inv_two_sigma_sq = 1.0f / (2.0f * sigma * sigma);
        kernel_raw(r) = exp(-off * off * inv_two_sigma_sq);
        RDom rk(0, 2 * R + 1, "rk");
        ksum() = 0.0f;
        ksum() += kernel_raw(rk);
        kernel(r) = kernel_raw(r) / ksum();

        blur_x(x, y) = 0.0f;
        blur_x(x, y) += alpha_b(x + rk - R, y) * kernel(rk);
        blur_y(x, y) = 0.0f;
        blur_y(x, y) += blur_x(x, y + rk - R) * kernel(rk);

        // Integer offset, then scale by opacity to a shadow coverage in [0, 1].
        Expr ox = cast<int>(round(dx)), oy = cast<int>(round(dy));
        Expr sh_a = clamp(blur_y(x - ox, y - oy) / 255.0f, 0.0f, 1.0f) * opacity;

        // Straight-alpha "source over shadow".
        Expr src_a = cast<float>(input(x, y, 3)) / 255.0f;
        Expr out_a = src_a + sh_a * (1.0f - src_a);
        Expr tint  = select(c == 0, tint_r, c == 1, tint_g, tint_b) / 255.0f;
        Expr src_c = cast<float>(input(x, y, c)) / 255.0f;
        Expr out_c = (src_c * src_a + tint * sh_a * (1.0f - src_a)) / max(out_a, 1e-6f);

        output(x, y, c) = cast<uint8_t>(
            clamp(select(c == 3, out_a, out_c), 0.0f, 1.0f) * 255.0f + 0.5f);

        fx::set_interleaved_layout(input);
        fx::set_interleaved_layout(output);
    }

    void schedule() {
        using namespace Halide;
        kernel_raw.compute_root();
        ksum.compute_root();
        kernel.compute_root();

        if (get_target().has_gpu_feature()) {
            // blur_x / blur_y inline into output (one K^2 kernel), as in gaussian.
            output.compute_root().reorder(c, x, y)
                  .gpu_tile(x, y, xo, yo, xi, yi, gpu_block_x, gpu_block_y,
                            TailStrategy::GuardWithIf);
        } else {
            constexpr int vec = 8, tile = 32;
            output.reorder(c, x, y)
                  .split(y, yo, yi, tile, TailStrategy::GuardWithIf).parallel(yo)
                  .vectorize(x, vec, TailStrategy::GuardWithIf);
            blur_x.compute_at(output, yo).store_at(output, yo)
                  .vectorize(x, vec, TailStrategy::GuardWithIf);
        }
        blur_x.update(0).unscheduled();
        blur_y.update(0).unscheduled();
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ShadowGenerator, shadow)
