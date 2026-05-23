// Effect registry: the single source of truth for what effects exist, what
// parameters they take, and which generated pipelines back them.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "HalideRuntime.h"

namespace fxcli {

// A named runtime parameter (a Halide Input<float> scalar). `lo`/`hi` are the inclusive validation bounds.
struct ParamSpec {
    const char* name;
    float       def;
    float       lo, hi;
};

// Marshals resolved, validated params into the concrete generated call.
using Invoke =
    std::function<int(halide_buffer_t* in, const std::vector<float>& params, halide_buffer_t* out)>;

struct Effect {
    std::string            name;
    std::vector<ParamSpec> params;
    Invoke                 cpu;   // required
    Invoke                 gpu;   // empty target => CPU-only (e.g. gamma)
};

const std::vector<Effect>& registry();
const Effect*              find_effect(const std::string& name);

}  // namespace fxcli
