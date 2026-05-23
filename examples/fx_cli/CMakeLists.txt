add_executable(fx_cli fx_cli.c)

target_link_libraries(fx_cli PRIVATE
    fx_gamma
    fx_gaussian_cpu
    fx_gaussian_gpu
    m
)

target_include_directories(fx_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/examples/third_party
)

# On WSL2 the NVIDIA driver lives in /usr/lib/wsl/lib.
if(EXISTS "/usr/lib/wsl/lib")
    set_target_properties(fx_cli PROPERTIES
        BUILD_RPATH "/usr/lib/wsl/lib"
        INSTALL_RPATH "/usr/lib/wsl/lib"
    )
endif()
