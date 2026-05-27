# libfx: the public shared library. Wraps the AOT effect libs (+ shared runtime)
# behind the C ABI in include/fx/fx.h. soname is libfx.so.1 (ABI generation 1),
# independent of the FX_VERSION_* release numbers in the header.
add_library(fx SHARED fx.cpp)

target_compile_features(fx PRIVATE cxx_std_17)
target_compile_definitions(fx PRIVATE FX_BUILDING)

set_target_properties(fx PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    SOVERSION 1
)

target_include_directories(fx PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Export only the public ABI; hide the statically-linked Halide kernels.
target_link_options(fx PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/fx.map"
)
set_target_properties(fx PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fx.map)

target_link_libraries(fx PRIVATE
    fx_gamma
    fx_gaussian_cpu
    fx_gaussian_gpu
    fx_kawase_cpu
    fx_kawase_gpu
    fx_shadow_cpu
    fx_shadow_gpu
    fx_color
    fx_rounded
    fx_runtime
)

# The GPU AOT libs dlopen libcuda.so, which on WSL2 lives in /usr/lib/wsl/lib.
# Bake it into the .so RPATH (no-op off WSL), same as examples/fx_cli.
if(EXISTS "/usr/lib/wsl/lib")
    set_target_properties(fx PROPERTIES
        BUILD_RPATH "/usr/lib/wsl/lib"
        INSTALL_RPATH "/usr/lib/wsl/lib"
    )
endif()
