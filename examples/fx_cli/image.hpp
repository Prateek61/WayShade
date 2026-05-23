// RAII wrapper around an 8-bit interleaved RGB(A) image: owns the host pixels

#pragma once

#include <cstdint>
#include <string>

#include "HalideRuntime.h"

namespace fxcli {

class Image {
public:
    static Image load(const std::string& path);  // throws std::runtime_error
    Image(int w, int h, int c);                   // uninitialized scratch

    ~Image();
    Image(Image&&) noexcept;
    Image& operator=(Image&&) noexcept;
    Image(const Image&)            = delete;
    Image& operator=(const Image&) = delete;

    void write_png(const std::string& path) const;  // throws std::runtime_error

    int      width() const { return w_; }
    int      height() const { return h_; }
    int      channels() const { return c_; }
    uint8_t* data() const { return px_; }

    // Build an interleaved descriptor over these pixels. `dims` must outlive
    // the returned buffer (it points into the caller's array).
    halide_buffer_t buffer(halide_dimension_t dims[3]) const;

private:
    Image() = default;
    void reset() noexcept;

    uint8_t* px_        = nullptr;
    bool     stb_owned_ = false;  // free via stbi_image_free vs free()
    int      w_ = 0, h_ = 0, c_ = 0;
};

}  // namespace fxcli
