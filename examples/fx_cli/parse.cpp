#include "parse.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "toml.hpp"

namespace fxcli {

namespace {

std::string to_g(float v) {
    char b[32];
    std::snprintf(b, sizeof b, "%g", v);
    return b;
}

Op make_op(const Effect* e) {
    Op op;
    op.effect = e;
    op.params.reserve(e->params.size());
    for (const ParamSpec& ps : e->params) op.params.push_back(ps.def);
    return op;
}

int param_index(const Effect* e, const std::string& name) {
    for (size_t i = 0; i < e->params.size(); ++i)
        if (name == e->params[i].name) return static_cast<int>(i);
    return -1;
}

bool try_float(const char* s, float& out) {
    char* end = nullptr;
    float v   = std::strtof(s, &end);
    if (end == s || *end != '\0') return false;  // empty or trailing junk
    out = v;
    return true;
}

void validate_op(const Op& op) {
    const Effect* e = op.effect;
    for (size_t i = 0; i < e->params.size(); ++i) {
        const ParamSpec& ps = e->params[i];
        float            v  = op.params[i];
        if (v >= ps.lo && v <= ps.hi) continue;  // also rejects NaN
        const bool pos_only = ps.lo == std::numeric_limits<float>::min() && std::isinf(ps.hi);
        if (pos_only)
            throw std::runtime_error(e->name + " parameter '" + ps.name +
                                     "' must be > 0 (got " + to_g(v) + ")");
        throw std::runtime_error(e->name + " parameter '" + ps.name + "' must be in [" +
                                 to_g(ps.lo) + ", " + to_g(ps.hi) + "] (got " + to_g(v) + ")");
    }
}

template <typename View>
std::optional<double> number_at(const View& nv) {
    if (auto d = nv.template value<double>()) return *d;
    if (auto i = nv.template value<int64_t>()) return static_cast<double>(*i);
    return std::nullopt;
}

}  // namespace

void parse_config_file(const std::string& path, std::vector<Op>& ops, Backend& backend) {
    toml::table tbl;
    try {
        tbl = toml::parse_file(path);
    } catch (const toml::parse_error& err) {
        throw std::runtime_error("config parse error in " + path + ": " +
                                 std::string(err.description()));
    }

    if (auto b = tbl["backend"].value<std::string>()) {
        if (*b == "cpu") backend = Backend::Cpu;
        else if (*b == "gpu") backend = Backend::Gpu;
        else throw std::runtime_error("config: backend must be \"cpu\" or \"gpu\" (got \"" + *b + "\")");
    }

    toml::array* arr = tbl["effect"].as_array();
    if (!arr || arr->empty())
        throw std::runtime_error("config: no [[effect]] entries in " + path);

    for (toml::node& node : *arr) {
        toml::table* et = node.as_table();
        if (!et) throw std::runtime_error("config: each effect must be a [[effect]] table");

        auto nm = (*et)["name"].value<std::string>();
        if (!nm) throw std::runtime_error("config: an [[effect]] is missing 'name'");
        const Effect* e = find_effect(*nm);
        if (!e) throw std::runtime_error("config: unknown effect '" + *nm + "'");

        Op op = make_op(e);
        for (size_t i = 0; i < e->params.size(); ++i)
            if (auto v = number_at((*et)[e->params[i].name])) op.params[i] = static_cast<float>(*v);

        // Typo guard: warn on keys that aren't 'name' or a known param.
        for (auto&& kv : *et) {
            std::string k(kv.first.str());
            if (k != "name" && param_index(e, k) < 0)
                std::fprintf(stderr, "fx_cli: warning: effect '%s' has no parameter '%s' (ignored)\n",
                             e->name.c_str(), k.c_str());
        }
        ops.push_back(std::move(op));
    }
}

Invocation parse_args(int argc, char** argv) {
    Invocation inv;

    // help/list can appear anywhere and short-circuit everything else.
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { inv.help = true; return inv; }
        if (a == "--list")              { inv.list = true; return inv; }
    }

    if (argc < 4)
        throw std::runtime_error("expected <input.png> <output.png> <effects...>\n\n" + usage(argv[0]));

    inv.in_path  = argv[1];
    inv.out_path = argv[2];

    std::string config_path;
    bool        gpu_flag = false;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--gpu") { gpu_flag = true; continue; }
        if (a == "--config" || a == "-c") {
            if (i + 1 >= argc) throw std::runtime_error("--config requires a file path");
            config_path = argv[++i];
            continue;
        }
        if (a.size() < 3 || a[0] != '-' || a[1] != '-')
            throw std::runtime_error("unexpected argument: " + a + "\n\n" + usage(argv[0]));

        std::string name = a.substr(2);

        if (const Effect* e = find_effect(name)) {
            // New op at defaults; optionally consume a bare-number first param
            // (preserves the legacy `--gamma 2.2` form).
            Op op = make_op(e);
            if (!e->params.empty() && i + 1 < argc) {
                float v;
                if (try_float(argv[i + 1], v)) { op.params[0] = v; ++i; }
            }
            inv.ops.push_back(std::move(op));
            continue;
        }

        // Otherwise: a named param `--<name> <value>` for the most-recent op.
        if (inv.ops.empty())
            throw std::runtime_error("--" + name + " given before any effect");
        Op&  cur = inv.ops.back();
        int  pi  = param_index(cur.effect, name);
        if (pi < 0)
            throw std::runtime_error("effect '" + cur.effect->name + "' has no parameter '" + name + "'");
        if (i + 1 >= argc)
            throw std::runtime_error("--" + name + " requires a value");
        float v;
        if (!try_float(argv[++i], v))
            throw std::runtime_error("--" + name + " value is not a number: " + argv[i]);
        cur.params[pi] = v;
    }

    if (!config_path.empty()) {
        if (!inv.ops.empty())
            throw std::runtime_error("--config cannot be combined with inline effect flags");
        parse_config_file(config_path, inv.ops, inv.backend);
    }

    if (gpu_flag) inv.backend = Backend::Gpu;  // CLI --gpu overrides config backend

    if (inv.ops.empty())
        throw std::runtime_error("no effects specified\n\n" + usage(argv[0]));

    for (const Op& op : inv.ops) validate_op(op);
    return inv;
}

std::string usage(const char* argv0) {
    std::string s = "usage: ";
    s += argv0;
    s += " <input.png> <output.png> (<effects...> | --config <file.toml>) [--gpu]\n"
         "  effects apply left-to-right; set a param with --<name> <value> or a bare value:\n"
         "    --gamma <v>              gamma correction (value > 0, default 1)\n"
         "    --gaussian --sigma <s>   separable Gaussian blur (sigma > 0, default 4)\n"
         "    --kawase --offset <o>    dual-Kawase blur (offset > 0, default 1)\n"
         "    --color [--brightness/--contrast/--saturation <v>]  color correction (default 1)\n"
         "    --rounded --radius <r>   rounded-corner alpha mask (radius >= 0, default 16)\n"
         "    --shadow [--sigma/--dx/--dy/--tint_r/--tint_g/--tint_b/--opacity <v>]  drop shadow\n"
         "  --config <file.toml>       read the effect chain from a TOML file instead\n"
         "  --gpu                      run GPU-capable passes on the GPU (CUDA)\n"
         "  --list                     list available effects and their parameters\n"
         "  --help                     show this message\n";
    return s;
}

void print_effects() {
    std::printf("available effects:\n");
    for (const Effect& e : registry()) {
        std::printf("  %-10s %-10s", e.name.c_str(), e.gpu ? "[cpu+gpu]" : "[cpu]");
        for (const ParamSpec& p : e.params) std::printf(" %s=%g", p.name, p.def);
        std::printf("\n");
    }
}

}  // namespace fxcli
