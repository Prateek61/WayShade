/* WayShade public C ABI. Opaque handles, explicit lifetimes, no C++ leakage. */

#ifndef FX_FX_H
#define FX_FX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FX_VERSION_MAJOR 0
#define FX_VERSION_MINOR 1
#define FX_VERSION_PATCH 0

#if defined(_WIN32)
#  if defined(FX_BUILDING)
#    define FX_API __declspec(dllexport)
#  else
#    define FX_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__)
#  define FX_API __attribute__((visibility("default")))
#else
#  define FX_API
#endif

/* Opaque handles. Each is heap-allocated by its _create call and released by the
   matching _destroy. A context must outlive every image and pipeline made from
   it. The handles are not thread-safe: use one context per thread. */
typedef struct fx_context_t  fx_context_t;
typedef struct fx_image_t    fx_image_t;
typedef struct fx_pipeline_t fx_pipeline_t;

typedef enum {
    FX_OK = 0,
    FX_ERR_INVALID_ARGUMENT,    /* NULL handle, bad dimensions, or out-of-range param */
    FX_ERR_OUT_OF_MEMORY,
    FX_ERR_UNSUPPORTED,         /* e.g. an alpha effect on a non-RGBA image */
    FX_ERR_DIMENSION_MISMATCH,  /* run output does not match the input */
    FX_ERR_BACKEND,             /* a Halide kernel or device call failed */
    FX_ERR_INTERNAL
} fx_status_t;

typedef enum {
    FX_BACKEND_CPU = 0,
    FX_BACKEND_GPU,   /* CUDA. Pointwise effects (gamma/color/rounded) fall back to CPU */
    FX_BACKEND_AUTO   /* GPU if available, else CPU */
} fx_backend_t;

/* "major.minor.patch" of the library, distinct from the .so soname. */
FX_API const char* fx_version(void);

/* Static name for a status code. Never NULL, not owned by the caller. */
FX_API const char* fx_status_string(fx_status_t status);

/* --- context --- */

FX_API fx_status_t fx_context_create(fx_backend_t backend, fx_context_t** out_ctx);
FX_API void        fx_context_destroy(fx_context_t* ctx);

/* Detail for the most recent failure on this context, valid until the next call
   on the same context. Returns "" when there is no error. */
FX_API const char* fx_context_last_error(const fx_context_t* ctx);

/* --- image: 8-bit interleaved RGB(A), channels innermost --- */

/* Allocate an uninitialized width*height*channels image. channels is 3 or 4. */
FX_API fx_status_t fx_image_create(fx_context_t* ctx, int width, int height,
                                   int channels, fx_image_t** out_img);

/* As fx_image_create, then copy width*height*channels bytes from pixels. */
FX_API fx_status_t fx_image_from_data(fx_context_t* ctx, int width, int height,
                                      int channels, const unsigned char* pixels,
                                      fx_image_t** out_img);

/* Writable host pixel buffer, interleaved, width*height*channels bytes. Valid
   until the image is destroyed. NULL if img is NULL. */
FX_API unsigned char* fx_image_data(fx_image_t* img);

/* Getters return -1 if img is NULL. */
FX_API int  fx_image_width(const fx_image_t* img);
FX_API int  fx_image_height(const fx_image_t* img);
FX_API int  fx_image_channels(const fx_image_t* img);
FX_API void fx_image_destroy(fx_image_t* img);

/* --- pipeline: an ordered effect chain, applied in the order effects are added --- */

FX_API fx_status_t fx_pipeline_create(fx_context_t* ctx, fx_pipeline_t** out_pipe);

/* Append one effect. Params are validated now: an out-of-range value returns
   FX_ERR_INVALID_ARGUMENT and sets the context last-error. */
FX_API fx_status_t fx_pipeline_gamma   (fx_pipeline_t* p, float value);   /* value > 0 */
FX_API fx_status_t fx_pipeline_gaussian(fx_pipeline_t* p, float sigma);   /* sigma > 0 */
FX_API fx_status_t fx_pipeline_kawase  (fx_pipeline_t* p, float offset);  /* offset > 0 */
FX_API fx_status_t fx_pipeline_color   (fx_pipeline_t* p, float brightness,
                                        float contrast, float saturation); /* each >= 0, 1 = identity */
FX_API fx_status_t fx_pipeline_rounded (fx_pipeline_t* p, float radius,
                                        float softness);                   /* needs RGBA */
FX_API fx_status_t fx_pipeline_shadow  (fx_pipeline_t* p, float sigma,
                                        float dx, float dy,
                                        unsigned char tint_r, unsigned char tint_g,
                                        unsigned char tint_b, float opacity); /* needs RGBA, opacity in [0,1] */

/* Drop all appended effects so the pipeline can be rebuilt and reused. */
FX_API fx_status_t fx_pipeline_clear(fx_pipeline_t* p);

/* Apply the chain. out must match in in width, height, and channels, and in is
   not modified. If any effect needs alpha, both images must be 4-channel or the
   call returns FX_ERR_UNSUPPORTED. */
FX_API fx_status_t fx_pipeline_run(fx_pipeline_t* p, const fx_image_t* in,
                                   fx_image_t* out);

FX_API void fx_pipeline_destroy(fx_pipeline_t* p);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* FX_FX_H */
