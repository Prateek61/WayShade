#include "pipeline.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace fxcli {

Image run_chain(const Image& input, const std::vector<Op>& ops, Backend backend) {
    const int    w = input.width(), h = input.height(), c = input.channels();
    const size_t nbytes = static_cast<size_t>(w) * h * c;

    // Ping-pong between two scratch buffers; `src` holds the current image.
    // Seed `src` from the input so the caller's image stays untouched.
    Image  a(w, h, c), b(w, h, c);
    std::memcpy(a.data(), input.data(), nbytes);
    Image* src = &a;
    Image* dst = &b;

    for (const Op& op : ops) {
        halide_dimension_t id[3], od[3];
        halide_buffer_t    in  = src->buffer(id);
        halide_buffer_t    out = dst->buffer(od);

        // GPU AOT copies host->device only when host_dirty is set; harmless on CPU.
        in.flags |= halide_buffer_flag_host_dirty;

        const bool    on_gpu = (backend == Backend::Gpu) && static_cast<bool>(op.effect->gpu);
        const Invoke& fn     = on_gpu ? op.effect->gpu : op.effect->cpu;

        int rc = fn(&in, op.params, &out);
        if (rc != halide_error_code_success)
            throw std::runtime_error("effect '" + op.effect->name +
                                     "' failed: halide error " + std::to_string(rc));

        if (on_gpu) {
            // Result is device-resident; pull it back, then free the alloc
            // unconditionally so an error can't leak it.
            int crc = halide_copy_to_host(nullptr, &out);
            halide_device_free(nullptr, &out);
            if (crc != halide_error_code_success)
                throw std::runtime_error("copy_to_host failed: halide error " +
                                         std::to_string(crc));
        }

        std::swap(src, dst);
    }

    return std::move(*src);  // result landed in *src after the final swap
}

}  // namespace fxcli
