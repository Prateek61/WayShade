use std::marker::PhantomData;
use std::ptr;

use fx_sys::{fx_image_t, fx_status_t};

use crate::error::last_error_string;
use crate::{Context, FxError, Result};

/// An 8-bit interleaved RGB(A) image owned by `libfx`, tied to the [`Context`]
/// that created it.
pub struct Image<'ctx> {
    ptr: *mut fx_image_t,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> Image<'ctx> {
    /// Allocate an uninitialized `width x height x channels` image. `channels`
    /// must be 3 (RGB) or 4 (RGBA).
    pub fn new(ctx: &'ctx Context, width: i32, height: i32, channels: i32) -> Result<Self> {
        let mut ptr: *mut fx_image_t = ptr::null_mut();
        // SAFETY: `ctx` is live; `out_img` is a valid out-pointer. On FX_OK the
        // callee writes a non-NULL handle we take ownership of.
        let status = unsafe { fx_sys::fx_image_create(ctx.as_ptr(), width, height, channels, &mut ptr) };
        Self::from_raw(status, ptr, ctx)
    }

    /// Allocate an image and copy `pixels` (exactly `width * height * channels`
    /// bytes) into it.
    pub fn from_data(
        ctx: &'ctx Context,
        width: i32,
        height: i32,
        channels: i32,
        pixels: &[u8],
    ) -> Result<Self> {
        if width <= 0 || height <= 0 || (channels != 3 && channels != 4) {
            return Err(FxError::InvalidArgument(format!(
                "invalid image dimensions {width}x{height}x{channels}"
            )));
        }
        let expected = width as usize * height as usize * channels as usize;
        if pixels.len() != expected {
            return Err(FxError::InvalidArgument(format!(
                "pixel buffer length {} does not match {width}x{height}x{channels} = {expected}",
                pixels.len()
            )));
        }
        let mut ptr: *mut fx_image_t = ptr::null_mut();
        // SAFETY: `pixels` is at least `expected` bytes (checked above); the C
        // side copies them, so the borrow need not outlive this call.
        let status = unsafe {
            fx_sys::fx_image_from_data(ctx.as_ptr(), width, height, channels, pixels.as_ptr(), &mut ptr)
        };
        Self::from_raw(status, ptr, ctx)
    }

    fn from_raw(status: fx_status_t, ptr: *mut fx_image_t, ctx: &'ctx Context) -> Result<Self> {
        if status != fx_status_t::FX_OK {
            return Err(FxError::from_status(status, last_error_string(ctx.as_ptr())));
        }
        if ptr.is_null() {
            return Err(FxError::Internal("image create returned OK but NULL".into()));
        }
        Ok(Image { ptr, _ctx: PhantomData })
    }

    /// Image width in pixels.
    pub fn width(&self) -> i32 {
        // SAFETY: `self.ptr` is a live, non-NULL handle for the image's lifetime.
        unsafe { fx_sys::fx_image_width(self.ptr) }
    }

    /// Image height in pixels.
    pub fn height(&self) -> i32 {
        unsafe { fx_sys::fx_image_height(self.ptr) }
    }

    /// Channel count, 3 or 4.
    pub fn channels(&self) -> i32 {
        unsafe { fx_sys::fx_image_channels(self.ptr) }
    }

    fn len(&self) -> usize {
        self.width().max(0) as usize * self.height().max(0) as usize * self.channels().max(0) as usize
    }

    /// The interleaved pixel bytes, `width * height * channels` long.
    pub fn data(&self) -> &[u8] {
        let len = self.len();
        // SAFETY: `fx_image_data` returns a pointer to exactly `len` writable
        // bytes valid for the image's lifetime; the shared borrow keeps it so.
        unsafe {
            let p = fx_sys::fx_image_data(self.ptr);
            std::slice::from_raw_parts(p, len)
        }
    }

    /// The interleaved pixel bytes, mutable.
    pub fn data_mut(&mut self) -> &mut [u8] {
        let len = self.len();
        // SAFETY: as `data`, and `&mut self` guarantees exclusive access.
        unsafe {
            let p = fx_sys::fx_image_data(self.ptr);
            std::slice::from_raw_parts_mut(p, len)
        }
    }

    pub(crate) fn as_ptr(&self) -> *const fx_image_t {
        self.ptr
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut fx_image_t {
        self.ptr
    }
}

impl std::fmt::Debug for Image<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("channels", &self.channels())
            .finish()
    }
}

impl Drop for Image<'_> {
    fn drop(&mut self) {
        // SAFETY: we own `self.ptr`, created by an `fx_image_*` call and never
        // copied or freed elsewhere, so this destroys it exactly once.
        unsafe { fx_sys::fx_image_destroy(self.ptr) }
    }
}

#[cfg(feature = "image")]
mod image_integration {
    use super::*;

    impl<'ctx> Image<'ctx> {
        /// Build an `Image` from an [`image::RgbaImage`] (4-channel).
        pub fn from_rgba8(ctx: &'ctx Context, img: &image::RgbaImage) -> Result<Self> {
            Image::from_data(ctx, img.width() as i32, img.height() as i32, 4, img.as_raw())
        }

        /// Build an `Image` from an [`image::RgbImage`] (3-channel).
        pub fn from_rgb8(ctx: &'ctx Context, img: &image::RgbImage) -> Result<Self> {
            Image::from_data(ctx, img.width() as i32, img.height() as i32, 3, img.as_raw())
        }

        /// Copy the pixels into an [`image::RgbaImage`]. Errors if the image is
        /// not 4-channel.
        pub fn to_rgba8(&self) -> Result<image::RgbaImage> {
            if self.channels() != 4 {
                return Err(FxError::Unsupported(format!(
                    "to_rgba8 needs a 4-channel image, have {}",
                    self.channels()
                )));
            }
            image::RgbaImage::from_raw(self.width() as u32, self.height() as u32, self.data().to_vec())
                .ok_or_else(|| FxError::Internal("RgbaImage::from_raw size mismatch".into()))
        }

        /// Copy the pixels into an [`image::RgbImage`]. Errors if the image is
        /// not 3-channel.
        pub fn to_rgb8(&self) -> Result<image::RgbImage> {
            if self.channels() != 3 {
                return Err(FxError::Unsupported(format!(
                    "to_rgb8 needs a 3-channel image, have {}",
                    self.channels()
                )));
            }
            image::RgbImage::from_raw(self.width() as u32, self.height() as u32, self.data().to_vec())
                .ok_or_else(|| FxError::Internal("RgbImage::from_raw size mismatch".into()))
        }
    }
}
