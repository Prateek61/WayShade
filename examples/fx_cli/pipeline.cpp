#include "pipeline.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace fxcli {

Image run_chain(const Image& input, const std::vector<Op>& ops, Backend backend) {
    const int w = input.width(), h = input.height(), c = input.channels();

    // shadow/rounded need an alpha channel. Promote the working buffers to RGBA
    // only when such an effect is in the chain, so pure RGB blur/gamma stays RGB.
    bool want_alpha = false;
    for (const Op& op : ops)
        if (op.effect->needs_alpha) { want_alpha = true; break; }
    const int wc = (want_alpha && c < 4) ? 4 : c;

    // Ping-pong between two scratch buffers; `src` holds the current image.
    // Seed `src` from the input so the caller's image stays untouched.
    Image a(w, h, wc), b(w, h, wc);
    if (wc == c) {
        std::memcpy(a.data(), input.data(), static_cast<size_t>(w) * h * c);
    } else {
        // Copy the present channels per pixel; fill the synthesized alpha opaque.
        const uint8_t* s = input.data();
        uint8_t*       d = a.data();
        for (size_t i = 0, n = static_cast<size_t>(w) * h; i < n; ++i) {
            int k = 0;
            for (; k < c && k < wc; ++k) d[i * wc + k] = s[i * c + k];
            for (; k < wc; ++k) d[i * wc + k] = (k == 3) ? 255 : 0;
        }
    }
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
