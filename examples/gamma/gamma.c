#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "HalideRuntime.h"
#include "fx_gamma.h"

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

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s input.png output.png <gamma>\n", argv[0]);
        return 1;
    }

    float gamma = (float)atof(argv[3]);
    if (gamma <= 0.0f) {
        fprintf(stderr, "gamma must be > 0 (got %g)\n", gamma);
        return 1;
    }

    // Load image through stb_image
    int w, h, c;
    uint8_t* in_pixels = stbi_load(argv[1], &w, &h, &c, 0);
    if (!in_pixels) {
        fprintf(stderr, "load failed: %s\n", argv[1]);
        return 1;
    }

    uint8_t* out_pixels = malloc((size_t)w * h * c);
    if (!out_pixels) { stbi_image_free(in_pixels); return 1; }

    halide_dimension_t in_dims[3], out_dims[3];
    halide_buffer_t in_buf, out_buf;
    fill_buffer_u8_interleaved(&in_buf,  in_dims,  in_pixels,  w, h, c);
    fill_buffer_u8_interleaved(&out_buf, out_dims, out_pixels, w, h, c);

    // Call the Halide pipeline.
    int rc = fx_gamma(&in_buf, gamma, &out_buf);
    if (rc != halide_error_code_success) {
        fprintf(stderr, "fx_gamma error %d\n", rc);
        stbi_image_free(in_pixels);
        free(out_pixels);
        return 1;
    }

    if (!stbi_write_png(argv[2], w, h, c, out_pixels, w * c)) {
        fprintf(stderr, "write failed: %s\n", argv[2]);
    }

    stbi_image_free(in_pixels);
    free(out_pixels);
    return 0;
}
