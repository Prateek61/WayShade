// libfx: the public C ABI on top of the Halide-generated kernels. All C++ stays
// behind the boundary; nothing here throws across the extern "C" functions.

#include "fx/fx.h"

#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "HalideRuntime.h"

#include "fx_color.h"
#include "fx_gamma.h"
#include "fx_gaussian_cpu.h"
#include "fx_gaussian_gpu.h"
#include "fx_kawase_cpu.h"
#include "fx_kawase_gpu.h"
#include "fx_rounded.h"
#include "fx_shadow_cpu.h"
#include "fx_shadow_gpu.h"

namespace {

struct Op {
    enum Kind { Gamma, Gaussian, Kawase, Color, Rounded, Shadow } kind;
    bool  needs_alpha;
    bool  has_gpu;
    float p[7];  // up to shadow's seven scalars, in declaration order
};

// Interleaved descriptor over host pixels; channels are unit-stride (see
// halide/fx_common.h and the buffer-layout convention in CLAUDE.md).
halide_buffer_t make_buffer(unsigned char* host, int w, int h, int c, halide_dimension_t d[3]) {
    d[0].min = 0; d[0].extent = w;     d[0].stride = c;     d[0].flags = 0;
    d[1].min = 0; d[1].extent = h;     d[1].stride = w * c; d[1].flags = 0;
    d[2].min = 0; d[2].extent = c;     d[2].stride = 1;     d[2].flags = 0;

    halide_buffer_t b;
    std::memset(&b, 0, sizeof(b));
    b.host       = host;
    b.dim        = d;
    b.dimensions = 3;
    b.type.code  = halide_type_uint;
    b.type.bits  = 8;
    b.type.lanes = 1;
    return b;
}

int invoke(const Op& op, bool gpu, halide_buffer_t* in, halide_buffer_t* out) {
    const float* p = op.p;
    switch (op.kind) {
        case Op::Gamma:    return fx_gamma(in, p[0], out);
        case Op::Gaussian: return gpu ? fx_gaussian_gpu(in, p[0], out) : fx_gaussian_cpu(in, p[0], out);
        case Op::Kawase:   return gpu ? fx_kawase_gpu(in, p[0], out)   : fx_kawase_cpu(in, p[0], out);
        case Op::Color:    return fx_color(in, p[0], p[1], p[2], out);
        case Op::Rounded:  return fx_rounded(in, p[0], p[1], out);
        case Op::Shadow:   return gpu ? fx_shadow_gpu(in, p[0], p[1], p[2], p[3], p[4], p[5], p[6], out)
                                      : fx_shadow_cpu(in, p[0], p[1], p[2], p[3], p[4], p[5], p[6], out);
    }
    return halide_error_code_internal_error;
}

bool is_finite(float v) { return std::isfinite(v); }

#define FX_STR2(x) #x
#define FX_STR(x) FX_STR2(x)
const char* const kVersion =
    FX_STR(FX_VERSION_MAJOR) "." FX_STR(FX_VERSION_MINOR) "." FX_STR(FX_VERSION_PATCH);

}  // namespace

struct fx_context_t {
    fx_backend_t backend;
    std::string  last_error;
};

struct fx_image_t {
    int                        w, h, c;
    std::vector<unsigned char> px;
};

struct fx_pipeline_t {
    fx_context_t*   ctx;
    std::vector<Op> ops;
};

namespace {

fx_status_t fail(fx_context_t* ctx, fx_status_t s, std::string msg) {
    if (ctx) ctx->last_error = std::move(msg);
    return s;
}

// Append a validated op; reused by every fx_pipeline_<effect>.
fx_status_t push(fx_pipeline_t* p, const Op& op) {
    try {
        p->ops.push_back(op);
    } catch (...) {
        return fail(p->ctx, FX_ERR_OUT_OF_MEMORY, "out of memory");
    }
    return FX_OK;
}

}  // namespace

extern "C" {

const char* fx_version(void) { return kVersion; }

const char* fx_status_string(fx_status_t status) {
    switch (status) {
        case FX_OK:                    return "ok";
        case FX_ERR_INVALID_ARGUMENT:  return "invalid argument";
        case FX_ERR_OUT_OF_MEMORY:     return "out of memory";
        case FX_ERR_UNSUPPORTED:       return "unsupported";
        case FX_ERR_DIMENSION_MISMATCH:return "dimension mismatch";
        case FX_ERR_BACKEND:           return "backend failure";
        case FX_ERR_INTERNAL:          return "internal error";
    }
    return "unknown status";
}

/* --- context --- */

fx_status_t fx_context_create(fx_backend_t backend, fx_context_t** out_ctx) {
    if (!out_ctx) return FX_ERR_INVALID_ARGUMENT;
    if (backend != FX_BACKEND_CPU && backend != FX_BACKEND_GPU && backend != FX_BACKEND_AUTO)
        return FX_ERR_INVALID_ARGUMENT;
    fx_context_t* ctx = new (std::nothrow) fx_context_t;
    if (!ctx) return FX_ERR_OUT_OF_MEMORY;
    ctx->backend = backend;
    *out_ctx     = ctx;
    return FX_OK;
}

void fx_context_destroy(fx_context_t* ctx) { delete ctx; }

const char* fx_context_last_error(const fx_context_t* ctx) {
    return ctx ? ctx->last_error.c_str() : "";
}

/* --- image --- */

fx_status_t fx_image_create(fx_context_t* ctx, int width, int height,
                            int channels, fx_image_t** out_img) {
    if (!ctx || !out_img) return FX_ERR_INVALID_ARGUMENT;
    ctx->last_error.clear();
    if (width <= 0 || height <= 0 || (channels != 3 && channels != 4))
        return fail(ctx, FX_ERR_INVALID_ARGUMENT, "image dims must be positive and channels 3 or 4");

    fx_image_t* img = new (std::nothrow) fx_image_t;
    if (!img) return fail(ctx, FX_ERR_OUT_OF_MEMORY, "out of memory");
    img->w = width; img->h = height; img->c = channels;
    try {
        img->px.resize(static_cast<size_t>(width) * height * channels);
    } catch (...) {
        delete img;
        return fail(ctx, FX_ERR_OUT_OF_MEMORY, "out of memory");
    }
    *out_img = img;
    return FX_OK;
}

fx_status_t fx_image_from_data(fx_context_t* ctx, int width, int height,
                               int channels, const unsigned char* pixels,
                               fx_image_t** out_img) {
    if (!pixels) {
        if (ctx) ctx->last_error = "pixels is NULL";
        return FX_ERR_INVALID_ARGUMENT;
    }
    fx_status_t s = fx_image_create(ctx, width, height, channels, out_img);
    if (s != FX_OK) return s;
    std::memcpy((*out_img)->px.data(), pixels, (*out_img)->px.size());
    return FX_OK;
}

unsigned char* fx_image_data(fx_image_t* img) { return img ? img->px.data() : nullptr; }

int  fx_image_width(const fx_image_t* img)    { return img ? img->w : -1; }
int  fx_image_height(const fx_image_t* img)   { return img ? img->h : -1; }
int  fx_image_channels(const fx_image_t* img) { return img ? img->c : -1; }
void fx_image_destroy(fx_image_t* img)        { delete img; }

/* --- pipeline --- */

fx_status_t fx_pipeline_create(fx_context_t* ctx, fx_pipeline_t** out_pipe) {
    if (!ctx || !out_pipe) return FX_ERR_INVALID_ARGUMENT;
    ctx->last_error.clear();
    fx_pipeline_t* p = new (std::nothrow) fx_pipeline_t;
    if (!p) return fail(ctx, FX_ERR_OUT_OF_MEMORY, "out of memory");
    p->ctx    = ctx;
    *out_pipe = p;
    return FX_OK;
}

fx_status_t fx_pipeline_gamma(fx_pipeline_t* p, float value) {
    if (!p) return FX_ERR_INVALID_ARGUMENT;
    p->ctx->last_error.clear();
    if (!(value > 0.0f) || !is_finite(value))
        return fail(p->ctx, FX_ERR_INVALID_ARGUMENT, "gamma: value must be > 0");
    Op op{}; op.kind = Op::Gamma; op.p[0] = value;
    return push(p, op);
}

fx_status_t fx_pipeline_gaussian(fx_pipeline_t* p, float sigma) {
    if (!p) return FX_ERR_INVALID_ARGUMENT;
    p->ctx->last_error.clear();
    if (!(sigma > 0.0f) || !is_finite(sigma))
        return fail(p->ctx, FX_ERR_INVALID_ARGUMENT, "gaussian: sigma must be > 0");
    Op op{}; op.kind = Op::Gaussian; op.has_gpu = true; op.p[0] = sigma;
    return push(p, op);
}

fx_status_t fx_pipeline_kawase(fx_pipeline_t* p, float offset) {
    if (!p) return FX_ERR_INVALID_ARGUMENT;
    p->ctx->last_error.clear();
    if (!(offset > 0.0f) || !is_finite(offset))
        return fail(p->ctx, FX_ERR_INVALID_ARGUMENT, "kawase: offset must be > 0");
    Op op{}; op.kind = Op::Kawase; op.has_gpu = true; op.p[0] = offset;
    return push(p, op);
}

fx_status_t fx_pipeline_color(fx_pipeline_t* p, float brightness, float contrast, float saturation) {
    if (!p) return FX_ERR_INVALID_ARGUMENT;
    p->ctx->last_error.clear();
    if (!(brightness >= 0.0f && contrast >= 0.0f && saturation >= 0.0f) ||
        !is_finite(brightness) || !is_finite(contrast) || !is_finite(saturation))
        return fail(p->ctx, FX_ERR_INVALID_ARGUMENT, "color: brightness/contrast/saturation must be >= 0");
    Op op{}; op.kind = Op::Color; op.p[0] = brightness; op.p[1] = contrast; op.p[2] = saturation;
    return push(p, op);
}

fx_status_t fx_pipeline_rounded(fx_pipeline_t* p, float radius, float softness) {
    if (!p) return FX_ERR_INVALID_ARGUMENT;
    p->ctx->last_error.clear();
    if (!(radius >= 0.0f) || !is_finite(radius))
        return fail(p->ctx, FX_ERR_INVALID_ARGUMENT, "rounded: radius must be >= 0");
    if (!(softness > 0.0f) || !is_finite(softness))
        return fail(p->ctx, FX_ERR_INVALID_ARGUMENT, "rounded: softness must be > 0");
    Op op{}; op.kind = Op::Rounded; op.needs_alpha = true; op.p[0] = radius; op.p[1] = softness;
    return push(p, op);
}

fx_status_t fx_pipeline_shadow(fx_pipeline_t* p, float sigma, float dx, float dy,
                               unsigned char tint_r, unsigned char tint_g,
                               unsigned char tint_b, float opacity) {
    if (!p) return FX_ERR_INVALID_ARGUMENT;
    p->ctx->last_error.clear();
    if (!(sigma > 0.0f) || !is_finite(sigma))
        return fail(p->ctx, FX_ERR_INVALID_ARGUMENT, "shadow: sigma must be > 0");
    if (!is_finite(dx) || !is_finite(dy))
        return fail(p->ctx, FX_ERR_INVALID_ARGUMENT, "shadow: dx/dy must be finite");
    if (!(opacity >= 0.0f && opacity <= 1.0f))
        return fail(p->ctx, FX_ERR_INVALID_ARGUMENT, "shadow: opacity must be in [0, 1]");
    Op op{}; op.kind = Op::Shadow; op.needs_alpha = true; op.has_gpu = true;
    op.p[0] = sigma; op.p[1] = dx; op.p[2] = dy;
    op.p[3] = tint_r; op.p[4] = tint_g; op.p[5] = tint_b; op.p[6] = opacity;
    return push(p, op);
}

fx_status_t fx_pipeline_clear(fx_pipeline_t* p) {
    if (!p) return FX_ERR_INVALID_ARGUMENT;
    p->ctx->last_error.clear();
    p->ops.clear();
    return FX_OK;
}

fx_status_t fx_pipeline_run(fx_pipeline_t* p, const fx_image_t* in, fx_image_t* out) {
    if (!p || !in || !out) return FX_ERR_INVALID_ARGUMENT;
    fx_context_t* ctx = p->ctx;
    ctx->last_error.clear();

    if (in->w != out->w || in->h != out->h || in->c != out->c)
        return fail(ctx, FX_ERR_DIMENSION_MISMATCH, "output must match input width, height, and channels");

    bool need_alpha = false;
    for (const Op& op : p->ops)
        if (op.needs_alpha) { need_alpha = true; break; }
    if (need_alpha && in->c != 4)
        return fail(ctx, FX_ERR_UNSUPPORTED, "shadow/rounded require a 4-channel RGBA image");

    const int    w = in->w, h = in->h, c = in->c;
    const size_t bytes = static_cast<size_t>(w) * h * c;

    if (p->ops.empty()) {  // identity chain: straight copy
        std::memcpy(out->px.data(), in->px.data(), bytes);
        return FX_OK;
    }

    const size_t N       = p->ops.size();
    const bool   gpu_ctx = (ctx->backend != FX_BACKEND_CPU);

    // One scratch buffer; ping-pong arranged by parity so the final op writes
    // straight into the caller's `out`. `in` is read-only throughout.
    std::vector<unsigned char> scratch;
    if (N >= 2) {
        try { scratch.resize(bytes); }
        catch (...) { return fail(ctx, FX_ERR_OUT_OF_MEMORY, "out of memory"); }
    }

    const unsigned char* cur_src = in->px.data();
    for (size_t i = 0; i < N; ++i) {
        unsigned char* dst_host = (((N - 1 - i) & 1u) == 0) ? out->px.data() : scratch.data();
        const Op&      op       = p->ops[i];
        const bool     on_gpu   = gpu_ctx && op.has_gpu;

        halide_dimension_t id[3], od[3];
        halide_buffer_t inb  = make_buffer(const_cast<unsigned char*>(cur_src), w, h, c, id);
        halide_buffer_t outb = make_buffer(dst_host, w, h, c, od);

        // GPU AOT copies host->device only when host_dirty is set; no-op on CPU.
        inb.flags |= halide_buffer_flag_host_dirty;

        int rc = invoke(op, on_gpu, &inb, &outb);
        if (rc != halide_error_code_success) {
            if (on_gpu) { halide_device_free(nullptr, &outb); halide_device_free(nullptr, &inb); }
            return fail(ctx, FX_ERR_BACKEND, "effect failed: halide error " + std::to_string(rc));
        }

        if (on_gpu) {
            // Result is device-resident: pull it back, then free both device
            // allocs so a per-frame loop does not leak VRAM.
            int crc = halide_copy_to_host(nullptr, &outb);
            halide_device_free(nullptr, &outb);
            halide_device_free(nullptr, &inb);
            if (crc != halide_error_code_success)
                return fail(ctx, FX_ERR_BACKEND, "copy_to_host failed: halide error " + std::to_string(crc));
        }

        cur_src = dst_host;
    }
    return FX_OK;
}

void fx_pipeline_destroy(fx_pipeline_t* p) { delete p; }

}  // extern "C"
