add_executable(gamma gamma.c)

target_link_libraries(gamma PRIVATE fx_gamma m)

target_include_directories(gamma PRIVATE
    ${CMAKE_SOURCE_DIR}/examples/third_party
)
