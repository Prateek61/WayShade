#include "Halide.h"
#include "fx_common.h"

namespace {

class GaussianGenerator : public Halide::Generator<GaussianGenerator> {
public:
    // Compile-time bounds. The 1D kernel has length 2*max_radius + 1.
    GeneratorParam<int> max_radius{"max_radius", 32};
    
    // GPU block sizes, exposed as GeneratorParams, can be tuned from the CMake side via PARAMS.
    GeneratorParam<int> gpu_block_x{"gpu_block_x", 32};
    GeneratorParam<int> gpu_block_y{"gpu_block_y", 8};

    Input<Halide::Buffer<uint8_t, 3>>  input{"input"};
    Input<float>                       sigma{"sigma", 1.0f};
    Output<Halide::Buffer<uint8_t, 3>> output{"output"};

    Halide::Var x{"x"}, y{"y"}, c{"c"};
    Halide::Var xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};
    Halide::Func bounded{"bounded"}, in_f{"in_f"};
    Halide::Func kernel_raw{"kernel_raw"}, ksum{"ksum"}, kernel{"kernel"};
    Halide::Func blur_x{"blur_x"}, blur_y{"blur_y"};
    Halide::Var r{"r"};

    void generate() {
        using namespace Halide;

        const int R = max_radius;

        // Edge-replicate boundary. Matches scipy mode='nearest' and Pillow.
        bounded = BoundaryConditions::repeat_edge(input);
        in_f(x, y, c) = cast<float>(bounded(x, y, c));

        // 1D Gaussian kernel sampled at integer offsets r in [0, 2R].
        // Offset from center is (r - R) in [-R, R].
        Expr offset = cast<float>(r - R);
        Expr inv_two_sigma_sq = 1.0f / (2.0f * sigma * sigma);
        kernel_raw(r) = exp(-offset * offset * inv_two_sigma_sq);

        // Pre-normalize the kernel.
        RDom rk(0, 2 * R + 1, "rk");
        ksum() = 0.0f;
        ksum() += kernel_raw(rk);
        kernel(r) = kernel_raw(r) / ksum();

        // Horizontal pass.
        blur_x(x, y, c) = 0.0f;
        blur_x(x, y, c) += in_f(x + rk - R, y, c) * kernel(rk);

        // Vertical pass.
        blur_y(x, y, c) = 0.0f;
        blur_y(x, y, c) += blur_x(x, y + rk - R, c) * kernel(rk);

        output(x, y, c) = cast<uint8_t>(clamp(blur_y(x, y, c), 0.0f, 255.0f) + 0.5f);

        fx::set_interleaved_layout(input);
        fx::set_interleaved_layout(output);
    }

    void schedule() {
        using namespace Halide;

        kernel_raw.compute_root();
        ksum.compute_root();
        kernel.compute_root();

        // GuardWithIf on every split/vectorize/gpu_tile: default ShiftInwards
        // places the last tile at extent-block, which goes negative when the
        // image is smaller than the block (1x1 / 7x5 etc.) and aborts on access.
        if (get_target().has_gpu_feature()) {
            // GPU: each thread does the full 2D recompute.
            output.gpu_tile(x, y, xo, yo, xi, yi,
                            gpu_block_x, gpu_block_y, TailStrategy::GuardWithIf);
        } else {
            // CPU: parallel over y-tiles, vectorize x by 8 (AVX2 float width).
            const int vec  = 8;
            const int tile = 32;

            output
                .reorder(c, x, y)
                .split(y, yo, yi, tile, TailStrategy::GuardWithIf)
                .parallel(yo)
                .vectorize(x, vec, TailStrategy::GuardWithIf);

            blur_x
                .compute_at(output, yo)
                .store_at(output, yo)
                .reorder(c, x, y)
                .vectorize(x, vec, TailStrategy::GuardWithIf);
        }

        // Inlined RDom updates
        blur_x.update(0).unscheduled();
        blur_y.update(0).unscheduled();
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(GaussianGenerator, gaussian)
