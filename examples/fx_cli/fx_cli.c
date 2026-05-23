// Unified CLI for the WayShade effects library. Loads a PNG with stb_image,
// applies one or more effects left-to-right (the order they appear on the
// command line), and writes the result.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "HalideRuntime.h"
#include "fx_gamma.h"
#include "fx_gaussian_cpu.h"
#include "fx_gaussian_gpu.h"

#define MAX_OPS 32

typedef enum { OP_GAMMA, OP_GAUSSIAN } op_kind;
typedef struct {
    op_kind kind;
    float   param;   // gamma value, or Gaussian sigma
} op_t;

// Image loaded through stb_image to halide buffer descriptor.
static void fill_buffer_u8_interleaved(
    halide_buffer_t* buf, halide_dimension_t dims[3],
    uint8_t* data, int w, int h, int c)
{
    dims[0].min = 0; dims[0].extent = w; dims[0].stride = c;
    dims[1].min = 0; dims[1].extent = h; dims[1].stride = w * c;
    dims[2].min = 0; dims[2].extent = c; dims[2].stride = 1;

    memset(buf, 0, sizeof(*buf));
    buf->host = data;
    buf->dimensions = 3;
    buf->dim = dims;
    buf->type.code = halide_type_uint;
    buf->type.bits = 8;
    buf->type.lanes = 1;
}

static void usage(const char* argv0) {
    fprintf(stderr,
        "usage: %s <input.png> <output.png> <effects...> [--gpu]\n"
        "  effects (applied left-to-right):\n"
        "    --gamma <g>             gamma correction (g > 0)\n"
        "    --gaussian --sigma <s>  separable Gaussian blur (s > 0)\n"
        "  modifiers:\n"
        "    --gpu                   run Gaussian passes on the GPU (CUDA)\n",
        argv0);
}

int main(int argc, char** argv) {
    if (argc < 4) { usage(argv[0]); return 1; }

    const char* in_path  = argv[1];
    const char* out_path = argv[2];

    op_t ops[MAX_OPS];
    int  n_ops   = 0;
    int  use_gpu = 0;

    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) {
            if (n_ops >= MAX_OPS) { fprintf(stderr, "too many effects\n"); return 1; }
            ops[n_ops].kind = OP_GAMMA;
            ops[n_ops].param = (float)atof(argv[++i]);
            n_ops++;
        } else if (strcmp(argv[i], "--gaussian") == 0) {
            if (n_ops >= MAX_OPS) { fprintf(stderr, "too many effects\n"); return 1; }
            ops[n_ops].kind = OP_GAUSSIAN;
            ops[n_ops].param = -1.0f;   // sigma, filled in by a later --sigma
            n_ops++;
        } else if (strcmp(argv[i], "--sigma") == 0 && i + 1 < argc) {
            // Attach to the most recent --gaussian.
            int j = n_ops - 1;
            while (j >= 0 && ops[j].kind != OP_GAUSSIAN) j--;
            if (j < 0) {
                fprintf(stderr, "--sigma with no preceding --gaussian\n"); return 1;
            }
            ops[j].param = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = 1;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (n_ops == 0) {
        fprintf(stderr, "must specify at least one effect\n");
        usage(argv[0]);
        return 1;
    }
    for (int k = 0; k < n_ops; ++k) {
        if (ops[k].kind == OP_GAMMA && ops[k].param <= 0.0f) {
            fprintf(stderr, "gamma must be > 0 (got %g)\n", ops[k].param); return 1;
        }
        if (ops[k].kind == OP_GAUSSIAN && ops[k].param <= 0.0f) {
            fprintf(stderr, "--gaussian requires --sigma > 0\n"); return 1;
        }
    }

    int w, h, c;
    uint8_t* pixels = stbi_load(in_path, &w, &h, &c, 0);
    if (!pixels) { fprintf(stderr, "load failed: %s\n", in_path); return 1; }

    // Ping-pong between the stb_image buffer and one scratch buffer. `src`
    // holds the current image; each op writes into `dst`, then we swap.
    size_t nbytes = (size_t)w * h * c;
    uint8_t* scratch = (uint8_t*)malloc(nbytes);
    if (!scratch) { stbi_image_free(pixels); return 1; }
    uint8_t* src = pixels;
    uint8_t* dst = scratch;

    int failed = 0;
    for (int k = 0; k < n_ops && !failed; ++k) {
        halide_dimension_t in_dims[3], out_dims[3];
        halide_buffer_t in_buf, out_buf;
        fill_buffer_u8_interleaved(&in_buf,  in_dims,  src, w, h, c);
        fill_buffer_u8_interleaved(&out_buf, out_dims, dst, w, h, c);

        // GPU AOT pipelines copy host→device only when host_dirty is set
        in_buf.flags |= halide_buffer_flag_host_dirty;

        int rc = 0, on_gpu = 0;
        if (ops[k].kind == OP_GAMMA) {
            rc = fx_gamma(&in_buf, ops[k].param, &out_buf);
        } else if (use_gpu) {
            rc = fx_gaussian_gpu(&in_buf, ops[k].param, &out_buf);
            on_gpu = 1;
        } else {
            rc = fx_gaussian_cpu(&in_buf, ops[k].param, &out_buf);
        }

        if (rc != halide_error_code_success) {
            fprintf(stderr, "halide pipeline error %d\n", rc);
            failed = 1;
            break;
        }

        // GPU pipelines leave the result on-device (device_dirty=true). Pull it back to host.
        if (on_gpu) {
            int crc = halide_copy_to_host(NULL, &out_buf);
            if (crc != halide_error_code_success) {
                fprintf(stderr, "halide_copy_to_host error %d\n", crc);
                failed = 1;
                break;
            }
            halide_device_free(NULL, &out_buf);
        }

        uint8_t* tmp = src; src = dst; dst = tmp;  // result is now in src
    }

    if (!failed && !stbi_write_png(out_path, w, h, c, src, w * c)) {
        fprintf(stderr, "write failed: %s\n", out_path);
        failed = 1;
    }

    stbi_image_free(pixels);
    free(scratch);
    return failed ? 1 : 0;
}
