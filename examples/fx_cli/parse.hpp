// The two front-ends that produce an effect chain: command-line flags and a
// TOML config file. Resolve + Validate params against the registry.

#pragma once

#include <string>
#include <vector>

#include "pipeline.hpp"

namespace fxcli {

struct Invocation {
    std::string     in_path, out_path;
    std::vector<Op> ops;
    Backend         backend = Backend::Cpu;
    bool            help = false, list = false;  // print-and-exit requests
};

// Parse argv into an Invocation. Throws std::runtime_error on bad input.
Invocation parse_args(int argc, char** argv);

// Parse a TOML chain file: fills `ops` and (if the file sets it) `backend`.
void parse_config_file(const std::string& path, std::vector<Op>& ops, Backend& backend);

std::string usage(const char* argv0);
void        print_effects();  // for --list

}  // namespace fxcli
