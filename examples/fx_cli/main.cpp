// Unified CLI for the WayShade effects library. Loads a PNG, applies a chain
// of effects (from command-line flags or a TOML config) left-to-right, writes
// the result.

#include <cstdio>
#include <exception>

#include "image.hpp"
#include "parse.hpp"
#include "pipeline.hpp"

int main(int argc, char** argv) {
    using namespace fxcli;
    try {
        Invocation inv = parse_args(argc, argv);
        if (inv.help) { std::fputs(usage(argv[0]).c_str(), stdout); return 0; }
        if (inv.list) { print_effects(); return 0; }

        Image input  = Image::load(inv.in_path);
        Image result = run_chain(input, inv.ops, inv.backend);
        result.write_png(inv.out_path);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fx_cli: %s\n", e.what());
        return 1;
    }
}
