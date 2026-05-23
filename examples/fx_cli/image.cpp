#include "image.hpp"

#include <cstdlib>
#include <stdexcept>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fxcli {

Image Image::load(const std::string& path) {
    int    w, h, c;
    uint8_t* px = stbi_load(path.c_str(), &w, &h, &c, 0);
    if (!px) throw std::runtime_error("load failed: " + path);
    Image img;
    img.px_        = px;
    img.stb_owned_ = true;
    img.w_ = w; img.h_ = h; img.c_ = c;
    return img;
}

Image::Image(int w, int h, int c) : w_(w), h_(h), c_(c) {
    px_ = static_cast<uint8_t*>(std::malloc(static_cast<size_t>(w) * h * c));
    if (!px_) throw std::runtime_error("out of memory allocating scratch image");
}

void Image::reset() noexcept {
    if (px_) {
        if (stb_owned_) stbi_image_free(px_);
        else            std::free(px_);
    }
    px_ = nullptr;
    stb_owned_ = false;
    w_ = h_ = c_ = 0;
}

Image::~Image() { reset(); }

Image::Image(Image&& o) noexcept
    : px_(o.px_), stb_owned_(o.stb_owned_), w_(o.w_), h_(o.h_), c_(o.c_) {
    o.px_ = nullptr;
    o.stb_owned_ = false;
    o.w_ = o.h_ = o.c_ = 0;
}

Image& Image::operator=(Image&& o) noexcept {
    if (this != &o) {
        reset();
        px_ = o.px_; stb_owned_ = o.stb_owned_;
        w_ = o.w_; h_ = o.h_; c_ = o.c_;
        o.px_ = nullptr; o.stb_owned_ = false;
        o.w_ = o.h_ = o.c_ = 0;
    }
    return *this;
}

void Image::write_png(const std::string& path) const {
    if (!stbi_write_png(path.c_str(), w_, h_, c_, px_, w_ * c_))
        throw std::runtime_error("write failed: " + path);
}

halide_buffer_t Image::buffer(halide_dimension_t dims[3]) const {
    // Interleaved: channels are unit-stride, x-stride = channel count.
    dims[0].min = 0; dims[0].extent = w_; dims[0].stride = c_;
    dims[1].min = 0; dims[1].extent = h_; dims[1].stride = w_ * c_;
    dims[2].min = 0; dims[2].extent = c_; dims[2].stride = 1;

    halide_buffer_t buf{};
    buf.host       = px_;
    buf.dim        = dims;
    buf.dimensions = 3;
    buf.type.code  = halide_type_uint;
    buf.type.bits  = 8;
    buf.type.lanes = 1;
    return buf;
}

}  // namespace fxcli
