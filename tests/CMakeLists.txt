# Pure-C ABI exerciser. -pedantic catches any C++ leakage in fx/fx.h.
add_executable(fx_capi_test fx_capi_test.c)

set_target_properties(fx_capi_test PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)
target_compile_options(fx_capi_test PRIVATE -Wall -Wextra -pedantic)

target_link_libraries(fx_capi_test PRIVATE fx ${CMAKE_DL_LIBS})

add_test(NAME fx_capi_test COMMAND fx_capi_test)

# Running the test exercises the GPU path, whose AOT libs dlopen libcuda.so.
if(EXISTS "/usr/lib/wsl/lib")
    set_target_properties(fx_capi_test PROPERTIES
        BUILD_RPATH "/usr/lib/wsl/lib"
        INSTALL_RPATH "/usr/lib/wsl/lib"
    )
endif()

# Reference-parity suite (tests/python): drives libfx via ctypes and compares its
# output to numpy / OpenCV references.
find_program(FX_VENV_PY NAMES python python3
             PATHS "${CMAKE_SOURCE_DIR}/.venv/bin" NO_DEFAULT_PATH)
if(FX_VENV_PY)
    add_test(NAME fx_reference_parity
             COMMAND "${FX_VENV_PY}" -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/python")
    set_tests_properties(fx_reference_parity PROPERTIES
        ENVIRONMENT "FX_LIB_DIR=$<TARGET_FILE_DIR:fx>")
endif()
