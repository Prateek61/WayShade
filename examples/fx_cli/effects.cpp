#include "effects.hpp"

#include <limits>

#include "fx_gamma.h"
#include "fx_gaussian_cpu.h"
#include "fx_gaussian_gpu.h"
#include "fx_kawase_cpu.h"
#include "fx_kawase_gpu.h"

namespace fxcli {

namespace {
constexpr float INF = std::numeric_limits<float>::infinity();
// Smallest positive normal — used as an inclusive lower bound to mean "> 0".
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
    };
    return r;
}

const Effect* find_effect(const std::string& name) {
    for (const auto& e : registry())
        if (e.name == name) return &e;
    return nullptr;
}

}  // namespace fxcli
