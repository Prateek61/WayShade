# Shared Halide runtime — every fx_* AOT lib uses it so we don't pull in
# multiple runtimes (with conflicting halide_malloc/etc. symbols) when the
# example links several effects.
add_halide_runtime(fx_runtime TARGETS host)

# Generators
add_halide_generator(gamma.generator
    SOURCES gamma_generator.cpp
)

add_halide_generator(gaussian.generator
    SOURCES gaussian_generator.cpp
)

add_halide_generator(kawase.generator
    SOURCES kawase_generator.cpp
)

# AOT libraries
add_halide_library(fx_gamma
    FROM gamma.generator
    GENERATOR gamma
    USE_RUNTIME fx_runtime
)

add_halide_library(fx_gaussian_cpu
    FROM gaussian.generator
    GENERATOR gaussian
    USE_RUNTIME fx_runtime
)

add_halide_library(fx_gaussian_gpu
    FROM gaussian.generator
    GENERATOR gaussian
    FEATURES cuda
    USE_RUNTIME fx_runtime
    PARAMS gpu_block_x=32 gpu_block_y=8
)

add_halide_library(fx_kawase_cpu
    FROM kawase.generator
    GENERATOR kawase
    USE_RUNTIME fx_runtime
)

add_halide_library(fx_kawase_gpu
    FROM kawase.generator
    GENERATOR kawase
    FEATURES cuda
    USE_RUNTIME fx_runtime
    PARAMS gpu_block_x=32 gpu_block_y=8
)
