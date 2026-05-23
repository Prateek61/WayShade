// The effect chain: an ordered list of resolved ops, and the runner that
// applies them left-to-right through a ping-pong buffer.

#pragma once

#include <vector>

#include "effects.hpp"
#include "image.hpp"

namespace fxcli {

enum class Backend { Cpu, Gpu };

struct Op {
    const Effect*      effect;
    std::vector<float> params;  // resolved + validated, in ParamSpec order
};

// Apply `ops` to `input` and return the result.
Image run_chain(const Image& input, const std::vector<Op>& ops, Backend backend);

}  // namespace fxcli
