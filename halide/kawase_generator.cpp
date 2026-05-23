#include "Halide.h"
#include "fx_common.h"

#include <string>
#include <vector>

namespace {

// Dual-Kawase blur (Bjørge's dual filter, the variant KDE/KWin ships).
// Descend a mip pyramid with a 5-tap downsample, then ascend it with an
// 8-tap upsample.
class KawaseGenerator : public Halide::Generator<KawaseGenerator> {
public:
    // Pyramid depth = number of downsample passes (and the same number of upsample passes).
    GeneratorParam<int> passes{"passes", 3};

    // GPU block dims, tunable from CMake via PARAMS.
    GeneratorParam<int> gpu_block_x{"gpu_block_x", 32};
    GeneratorParam<int> gpu_block_y{"gpu_block_y", 8};

    Input<Halide::Buffer<uint8_t, 3>>  input{"input"};
    // KWin-style strength: scales the half-texel tap distance, so blur radius
    // is tunable per call without recompiling (radius grows ~ offset * 2^passes).
    Input<float>                       offset{"offset", 1.0f};
    Output<Halide::Buffer<uint8_t, 3>> output{"output"};

    Halide::Var x{"x"}, y{"y"}, c{"c"};
    Halide::Var xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};

    // Every materialized pyramid level (down1..N, up1..N-1); scheduled together.
    // up0 is folded into `output`, so it isn't here.
    std::vector<Halide::Func> levels;

    void generate() {
        using namespace Halide;
        const int N = passes;  // >= 1; set from CMake (default 3)

        // Edge-replicate + promote to float; safe to sample at any coordinate.
        Func bounded = BoundaryConditions::repeat_edge(input);
        Func in_f{"in_f"};
        in_f(x, y, c) = cast<float>(bounded(x, y, c));

        // Per-level extents, rounded up so a 1-px dimension never collapses.
        std::vector<Expr> w(N + 1), h(N + 1);
        w[0] = input.dim(0).extent();
        h[0] = input.dim(1).extent();
        for (int k = 1; k <= N; ++k) {
            w[k] = (w[k - 1] + 1) / 2;
            h[k] = (h[k - 1] + 1) / 2;
        }

        // Bilinear tap. `s` must already be edge-clamped so x0+1 / y0+1 are in range.
        auto bilin = [&](Func s, Expr fx, Expr fy) -> Expr {
            Expr x0 = cast<int>(floor(fx)), y0 = cast<int>(floor(fy));
            Expr tx = fx - cast<float>(x0), ty = fy - cast<float>(y0);
            Expr a = lerp(s(x0, y0, c),     s(x0 + 1, y0, c),     tx);
            Expr b = lerp(s(x0, y0 + 1, c), s(x0 + 1, y0 + 1, c), tx);
            return lerp(a, b, ty);
        };
        auto clamp_level = [&](Func f, Expr fw, Expr fh) -> Func {
            return BoundaryConditions::repeat_edge(f, {{0, fw}, {0, fh}, {Expr(), Expr()}});
        };

        // 5-tap downsample: center*4 + 4 diagonals, /8. Target pixel (x,y) maps
        // to source texel center (2x+0.5, 2y+0.5); diagonals sit a half-texel
        // (scaled by offset) away.
        auto downsample_expr = [&](Func s) -> Expr {
            Expr cx = 2.0f * x + 0.5f, cy = 2.0f * y + 0.5f;
            Expr d = 0.5f * offset;
            return (bilin(s, cx,     cy)     * 4.0f
                  + bilin(s, cx - d, cy - d)
                  + bilin(s, cx + d, cy + d)
                  + bilin(s, cx + d, cy - d)
                  + bilin(s, cx - d, cy + d)) * (1.0f / 8.0f);
        };

        // 8-tap upsample: 4 diagonals*2 + 4 axis*1, /12. Target pixel maps to
        // source texel center (0.5x-0.25, 0.5y-0.25); hp = half-texel, hp2 = one.
        auto upsample_expr = [&](Func s) -> Expr {
            Expr cx = 0.5f * x - 0.25f, cy = 0.5f * y - 0.25f;
            Expr hp = 0.5f * offset, hp2 = offset;
            return (bilin(s, cx - hp2, cy)
                  + bilin(s, cx - hp,  cy + hp)  * 2.0f
                  + bilin(s, cx,       cy + hp2)
                  + bilin(s, cx + hp,  cy + hp)  * 2.0f
                  + bilin(s, cx + hp2, cy)
                  + bilin(s, cx + hp,  cy - hp)  * 2.0f
                  + bilin(s, cx,       cy - hp2)
                  + bilin(s, cx - hp,  cy - hp)  * 2.0f) * (1.0f / 12.0f);
        };

        Func src = in_f;  // level 0, clamped via `bounded`
        for (int k = 1; k <= N; ++k) {
            Func down("down" + std::to_string(k));
            down(x, y, c) = downsample_expr(src);
            levels.push_back(down);
            src = clamp_level(down, w[k], h[k]);
        }
        for (int k = N - 1; k >= 0; --k) {
            if (k == 0) {
                // up0 is full-res — fold the cast-to-u8 in to skip a float buffer.
                output(x, y, c) =
                    cast<uint8_t>(clamp(upsample_expr(src), 0.0f, 255.0f) + 0.5f);
            } else {
                Func up("up" + std::to_string(k));
                up(x, y, c) = upsample_expr(src);
                levels.push_back(up);
                src = clamp_level(up, w[k], h[k]);
            }
        }

        fx::set_interleaved_layout(input);
        fx::set_interleaved_layout(output);
    }

    void schedule() {
        using namespace Halide;
        if (get_target().has_gpu_feature()) {
            // One kernel per level; intermediates stay device-resident between
            // passes (no host round-trips inside the pipeline).
            for (Func f : levels) {
                f.compute_root().reorder(c, x, y)
                 .gpu_tile(x, y, xo, yo, xi, yi, gpu_block_x, gpu_block_y,
                           TailStrategy::GuardWithIf);
            }
            output.compute_root().reorder(c, x, y)
                  .gpu_tile(x, y, xo, yo, xi, yi, gpu_block_x, gpu_block_y,
                            TailStrategy::GuardWithIf);
        } else {
            // Parallel over rows, vectorize x by AVX2 float width. GuardWithIf
            // because deep levels can be narrower than the vector.
            const int vec = 8;
            for (Func f : levels) {
                f.compute_root().reorder(c, x, y)
                 .parallel(y).vectorize(x, vec, TailStrategy::GuardWithIf);
            }
            output.compute_root().reorder(c, x, y)
                  .parallel(y).vectorize(x, vec, TailStrategy::GuardWithIf);
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(KawaseGenerator, kawase)
