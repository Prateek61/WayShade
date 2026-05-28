/* Pure-C exerciser for the libfx ABI. Builds as C11 -pedantic to prove the
   header carries no C++ leakage, and touches every public function. */

#include <stdio.h>
#include <stdlib.h>

#include <dlfcn.h>

#include "fx/fx.h"

static int failures = 0;

/* The GPU AOT kernels dlopen libcuda at first launch, aborting (not returning an
   error) when no driver is present.*/
static int gpu_available(void) {
    static const char* names[] = {
        "libcuda.so.1", "libcuda.so", "/usr/lib/wsl/lib/libcuda.so.1", NULL
    };
    int i;
    for (i = 0; names[i]; ++i) {
        void* h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (h) { dlclose(h); return 1; }
    }
    return 0;
}

#define CHECK(cond, msg)                                            \
    do {                                                            \
        if (!(cond)) { printf("  FAIL: %s\n", (msg)); ++failures; } \
        else         { printf("  ok:   %s\n", (msg)); }             \
    } while (0)

static void fill_gradient(unsigned char* p, int w, int h, int c) {
    int x, y;
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            unsigned char* px = p + ((size_t)y * w + x) * c;
            px[0] = (unsigned char)(w > 1 ? x * 255 / (w - 1) : 0);
            if (c > 1) px[1] = (unsigned char)(h > 1 ? y * 255 / (h - 1) : 0);
            if (c > 2) px[2] = (unsigned char)((x + y) & 255);
            if (c > 3) px[3] = 255;
        }
    }
}

static int max_abs_diff(const unsigned char* a, const unsigned char* b, size_t n) {
    int m = 0;
    size_t i;
    for (i = 0; i < n; ++i) {
        int d = (int)a[i] - (int)b[i];
        if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    const int W = 96, H = 64;
    fx_context_t* ctx = NULL;
    fx_status_t   s;
    int           i;

    printf("libfx version %s\n", fx_version());

    /* Every status code maps to a non-empty string. */
    printf("status strings:\n");
    for (i = FX_OK; i <= FX_ERR_INTERNAL; ++i) {
        const char* str = fx_status_string((fx_status_t)i);
        CHECK(str && str[0] != '\0', str ? str : "(null)");
    }

    /* --- context + image lifecycle --- */
    printf("context + image:\n");
    s = fx_context_create(FX_BACKEND_CPU, &ctx);
    CHECK(s == FX_OK && ctx != NULL, "create CPU context");
    CHECK(fx_context_last_error(ctx)[0] == '\0', "fresh context has empty last_error");

    {
        unsigned char* seed = (unsigned char*)malloc((size_t)W * H * 3);
        fx_image_t *in = NULL, *out = NULL;
        fill_gradient(seed, W, H, 3);

        s = fx_image_from_data(ctx, W, H, 3, seed, &in);
        CHECK(s == FX_OK, "fx_image_from_data (RGB)");
        CHECK(fx_image_width(in) == W && fx_image_height(in) == H &&
              fx_image_channels(in) == 3, "image getters report dims");
        CHECK(fx_image_data(in) != NULL, "fx_image_data is non-NULL");
        CHECK(max_abs_diff(fx_image_data(in), seed, (size_t)W * H * 3) == 0,
              "from_data copied pixels verbatim");

        s = fx_image_create(ctx, W, H, 3, &out);
        CHECK(s == FX_OK, "fx_image_create output (RGB)");

        /* --- pipeline: identity copy when empty --- */
        printf("pipeline (CPU):\n");
        {
            fx_pipeline_t* p = NULL;
            CHECK(fx_pipeline_create(ctx, &p) == FX_OK, "fx_pipeline_create");

            CHECK(fx_pipeline_run(p, in, out) == FX_OK, "run empty pipeline");
            CHECK(max_abs_diff(fx_image_data(in), fx_image_data(out),
                               (size_t)W * H * 3) == 0, "empty pipeline copies input");

            /* gamma 1.0 round-trips within float->u8 rounding (+/-1). */
            CHECK(fx_pipeline_gamma(p, 1.0f) == FX_OK, "add gamma 1.0");
            CHECK(fx_pipeline_run(p, in, out) == FX_OK, "run gamma 1.0");
            CHECK(max_abs_diff(fx_image_data(in), fx_image_data(out),
                               (size_t)W * H * 3) <= 1, "gamma 1.0 is near-identity");

            /* color identity (1,1,1) round-trips exactly. */
            CHECK(fx_pipeline_clear(p) == FX_OK, "fx_pipeline_clear");
            CHECK(fx_pipeline_color(p, 1.0f, 1.0f, 1.0f) == FX_OK, "add color identity");
            CHECK(fx_pipeline_run(p, in, out) == FX_OK, "run color identity");
            CHECK(max_abs_diff(fx_image_data(in), fx_image_data(out),
                               (size_t)W * H * 3) <= 1, "color identity round-trips");

            /* gaussian + kawase actually change the image. */
            fx_pipeline_clear(p);
            CHECK(fx_pipeline_gaussian(p, 4.0f) == FX_OK, "add gaussian sigma 4");
            CHECK(fx_pipeline_run(p, in, out) == FX_OK, "run gaussian");
            CHECK(max_abs_diff(fx_image_data(in), fx_image_data(out),
                               (size_t)W * H * 3) > 0, "gaussian changed pixels");

            fx_pipeline_clear(p);
            CHECK(fx_pipeline_kawase(p, 1.0f) == FX_OK, "add kawase offset 1");
            CHECK(fx_pipeline_run(p, in, out) == FX_OK, "run kawase");

            /* error: out-of-range param. */
            CHECK(fx_pipeline_gaussian(p, 0.0f) == FX_ERR_INVALID_ARGUMENT,
                  "gaussian sigma 0 rejected");
            CHECK(fx_context_last_error(ctx)[0] != '\0', "last_error set on bad param");

            /* error: alpha effect on an RGB image. */
            fx_pipeline_clear(p);
            fx_pipeline_rounded(p, 16.0f, 1.0f);
            CHECK(fx_pipeline_run(p, in, out) == FX_ERR_UNSUPPORTED,
                  "rounded on RGB rejected (needs RGBA)");

            fx_pipeline_destroy(p);
        }

        /* error: output dims mismatch. */
        {
            fx_pipeline_t* p   = NULL;
            fx_image_t*    bad = NULL;
            fx_pipeline_create(ctx, &p);
            fx_pipeline_gaussian(p, 2.0f);
            fx_image_create(ctx, W + 1, H, 3, &bad);
            CHECK(fx_pipeline_run(p, in, bad) == FX_ERR_DIMENSION_MISMATCH,
                  "size-mismatched output rejected");
            fx_image_destroy(bad);
            fx_pipeline_destroy(p);
        }

        fx_image_destroy(out);
        fx_image_destroy(in);
        free(seed);
    }

    /* --- RGBA effects: rounded drops corner alpha, shadow runs --- */
    printf("RGBA effects (CPU):\n");
    {
        unsigned char* seed = (unsigned char*)malloc((size_t)W * H * 4);
        fx_image_t *in = NULL, *out = NULL;
        fx_pipeline_t* p = NULL;
        fill_gradient(seed, W, H, 4); /* alpha = 255 everywhere */

        fx_image_from_data(ctx, W, H, 4, seed, &in);
        fx_image_create(ctx, W, H, 4, &out);
        fx_pipeline_create(ctx, &p);

        fx_pipeline_rounded(p, 24.0f, 1.0f);
        CHECK(fx_pipeline_run(p, in, out) == FX_OK, "run rounded on RGBA");
        CHECK(fx_image_data(out)[3] < 255, "rounded made the top-left corner transparent");

        fx_pipeline_clear(p);
        CHECK(fx_pipeline_shadow(p, 8.0f, 0.0f, 8.0f, 0, 0, 0, 0.5f) == FX_OK,
              "add shadow");
        CHECK(fx_pipeline_run(p, in, out) == FX_OK, "run shadow on RGBA");

        fx_pipeline_destroy(p);
        fx_image_destroy(out);
        fx_image_destroy(in);
        free(seed);
    }

    fx_context_destroy(ctx);

    /* --- GPU backend: matches CPU within float->u8 rounding; CPU fallback for
       pointwise effects inside a GPU chain --- */
    printf("GPU backend:\n");
    if (!gpu_available()) {
        printf("  skip: libcuda not loadable (no GPU/driver)\n");
    } else {
        fx_context_t *cpu = NULL, *gpu = NULL;
        unsigned char* seed = (unsigned char*)malloc((size_t)W * H * 3);
        fill_gradient(seed, W, H, 3);

        if (fx_context_create(FX_BACKEND_GPU, &gpu) == FX_OK &&
            fx_context_create(FX_BACKEND_CPU, &cpu) == FX_OK) {
            fx_image_t *in = NULL, *oc = NULL, *og = NULL;
            fx_pipeline_t *pc = NULL, *pg = NULL;
            fx_image_from_data(cpu, W, H, 3, seed, &in);
            fx_image_create(cpu, W, H, 3, &oc);
            fx_image_create(gpu, W, H, 3, &og);

            fx_pipeline_create(cpu, &pc);
            fx_pipeline_create(gpu, &pg);
            /* gaussian (has GPU) then color (CPU-only, falls back mid-chain). */
            fx_pipeline_gaussian(pc, 4.0f); fx_pipeline_color(pc, 1.1f, 1.0f, 1.0f);
            fx_pipeline_gaussian(pg, 4.0f); fx_pipeline_color(pg, 1.1f, 1.0f, 1.0f);

            CHECK(fx_pipeline_run(pc, in, oc) == FX_OK, "run chain on CPU");
            s = fx_pipeline_run(pg, in, og);
            CHECK(s == FX_OK, "run chain on GPU (with CPU fallback for color)");
            if (s == FX_OK)
                CHECK(max_abs_diff(fx_image_data(oc), fx_image_data(og),
                                   (size_t)W * H * 3) <= 1, "GPU matches CPU within +/-1");

            fx_pipeline_destroy(pg); fx_pipeline_destroy(pc);
            fx_image_destroy(og); fx_image_destroy(oc); fx_image_destroy(in);
        } else {
            printf("  skip: no GPU context (CUDA unavailable)\n");
        }
        fx_context_destroy(gpu);
        fx_context_destroy(cpu);
        free(seed);
    }

    printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
