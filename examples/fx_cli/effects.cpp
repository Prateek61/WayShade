#include "effects.hpp"

#include <limits>

#include "fx_color.h"
#include "fx_gamma.h"
#include "fx_gaussian_cpu.h"
#include "fx_gaussian_gpu.h"
#include "fx_kawase_cpu.h"
#include "fx_kawase_gpu.h"
#include "fx_rounded.h"
#include "fx_shadow_cpu.h"
#include "fx_shadow_gpu.h"

namespace fxcli {

namespace {
constexpr float INF = std::numeric_limits<float>::infinity();
// Smallest positive normal, used as an inclusive lower bound to mean "> 0".
constexpr float POS = std::numeric_limits<float>::min();
}  // namespace

const std::vector<Effect>& registry() {
    static const std::vector<Effect> r = {
        {"gamma",
         {{"value", 1.0f, POS, INF}},
         [](halide_buffer_t* in, const std::vector<float>& p, halide_buffer_t* out) {
             return fx_gamma(in, p[0], out);
         },
         {}},  // CPU-only: no fx_gamma_gpu exists

        {"gaussian",
         {{"sigma", 4.0f, POS, INF}},
         [](halide_buffer_t* in, const std::vector<float>& p, halide_buffer_t* out) {
             return fx_gaussian_cpu(in, p[0], out);
         },
         [](halide_buffer_t* in, const std::vector<float>& p, halide_buffer_t* out) {
             return fx_gaussian_gpu(in, p[0], out);
         }},

        {"kawase",
         {{"offset", 1.0f, POS, INF}},
         [](halide_buffer_t* in, const std::vector<float>& p, halide_buffer_t* out) {
             return fx_kawase_cpu(in, p[0], out);
         },
         [](halide_buffer_t* in, const std::vector<float>& p, halide_buffer_t* out) {
             return fx_kawase_gpu(in, p[0], out);
         }},

        {"color",
         {{"brightness", 1.0f, 0.0f, INF},
          {"contrast", 1.0f, 0.0f, INF},
          {"saturation", 1.0f, 0.0f, INF}},
         [](halide_buffer_t* in, const std::vector<float>& p, halide_buffer_t* out) {
             return fx_color(in, p[0], p[1], p[2], out);
         },
         {}},  // CPU-only

        {"rounded",
         {{"radius", 16.0f, 0.0f, INF},
          {"softness", 1.0f, POS, INF}},
         [](halide_buffer_t* in, const std::vector<float>& p, halide_buffer_t* out) {
             return fx_rounded(in, p[0], p[1], out);
         },
         {},     // CPU-only
         true},  // needs RGBA

        {"shadow",
         {{"sigma", 8.0f, POS, INF},
          {"dx", 0.0f, -INF, INF},
          {"dy", 8.0f, -INF, INF},
          {"tint_r", 0.0f, 0.0f, 255.0f},
          {"tint_g", 0.0f, 0.0f, 255.0f},
          {"tint_b", 0.0f, 0.0f, 255.0f},
          {"opacity", 0.5f, 0.0f, 1.0f}},
         [](halide_buffer_t* in, const std::vector<float>& p, halide_buffer_t* out) {
             return fx_shadow_cpu(in, p[0], p[1], p[2], p[3], p[4], p[5], p[6], out);
         },
         [](halide_buffer_t* in, const std::vector<float>& p, halide_buffer_t* out) {
             return fx_shadow_gpu(in, p[0], p[1], p[2], p[3], p[4], p[5], p[6], out);
         },
         true},  // needs RGBA
    };
    return r;
}

const Effect* find_effect(const std::string& name) {
    for (const auto& e : registry())
        if (e.name == name) return &e;
    return nullptr;
}

}  // namespace fxcli
