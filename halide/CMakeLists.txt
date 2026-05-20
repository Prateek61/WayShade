add_halide_generator(gamma.generator
    SOURCES gamma_generator.cpp
)

add_halide_library(fx_gamma
    FROM gamma.generator
    GENERATOR gamma
)
