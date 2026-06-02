//! Safe, idiomatic Rust bindings to `libfx`, the WayShade effects library.
//!
//! ```no_run
//! use fx::{Backend, Context, Image, Shadow};
//!
//! # fn main() -> Result<(), fx::FxError> {
//! let ctx = Context::new(Backend::Cpu)?;
//!
//! // RGBA so the alpha effects (shadow/rounded) are legal.
//! let pixels = vec![0u8; 256 * 256 * 4];
//! let input = Image::from_data(&ctx, 256, 256, 4, &pixels)?;
//! let mut output = Image::new(&ctx, 256, 256, 4)?;
//!
//! ctx.pipeline()
//!     .gaussian(8.0)
//!     .shadow(Shadow { dy: 12.0, ..Default::default() })
//!     .gamma(0.9)
//!     .run(&input, &mut output)?;
//! # Ok(()) }
//! ```
//!
//! # Lifetimes and threading
//!
//! [`Image`] and [`Pipeline`] borrow their [`Context`] (`Image<'ctx>`), so the
//! compiler enforces the ABI rule that a context outlives anything made from it.
//! The handles are neither `Send` nor `Sync`, so use one context per thread.
//!
//! # The unsafe boundary
//!
//! A few invariants keep this sound. Every handle comes from a `*_create` call
//! and is freed exactly once in `Drop` (the types aren't `Clone`/`Copy` and
//! never hand out their raw pointer). [`Image::data`] only exposes a slice of
//! `width * height * channels` bytes, and C strings are copied into owned
//! `String`s before the next call can invalidate them.
//!
//! # GPU caveat
//!
//! [`Backend::Gpu`] and [`Backend::Auto`] use CUDA. With no working CUDA driver
//! the Halide runtime **aborts the process** (an uncatchable `cuInit` failure),
//! so it is not a recoverable `Err`. The abort is not UB, so the API stays safe
//! in the Rust sense, but prefer [`Backend::Cpu`] unless a GPU is present.

mod error;
mod image;
mod pipeline;

pub use error::FxError;
pub use image::Image;
pub use pipeline::{Pipeline, Shadow};

use std::ffi::CStr;
use std::ptr;

/// Shorthand for `Result<T, FxError>`.
pub type Result<T> = std::result::Result<T, FxError>;

/// Which compute backend a [`Context`] dispatches effects to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CPU only.
    Cpu,
    /// CUDA. Pointwise effects (gamma/color/rounded) still run on the CPU.
    Gpu,
    /// GPU if available, else CPU. See the crate-level GPU caveat.
    Auto,
}

impl From<Backend> for fx_sys::fx_backend_t {
    fn from(b: Backend) -> Self {
        match b {
            Backend::Cpu => fx_sys::fx_backend_t::FX_BACKEND_CPU,
            Backend::Gpu => fx_sys::fx_backend_t::FX_BACKEND_GPU,
            Backend::Auto => fx_sys::fx_backend_t::FX_BACKEND_AUTO,
        }
    }
}

/// A `libfx` context: the root handle that owns the chosen [`Backend`] and from
/// which images and pipelines are created.
///
/// Not `Send`/`Sync`; one per thread.
pub struct Context {
    ptr: *mut fx_sys::fx_context_t,
}

impl Context {
    /// Create a context on `backend`.
    pub fn new(backend: Backend) -> Result<Self> {
        let mut ptr: *mut fx_sys::fx_context_t = ptr::null_mut();
        // SAFETY: out-pointer is valid; on FX_OK we own the handle written to it.
        let status = unsafe { fx_sys::fx_context_create(backend.into(), &mut ptr) };
        if status != fx_sys::fx_status_t::FX_OK {
            // No context yet, so there's nowhere to read an error detail from.
            return Err(FxError::from_status(status, String::new()));
        }
        if ptr.is_null() {
            return Err(FxError::Internal("fx_context_create returned OK but NULL".into()));
        }
        Ok(Context { ptr })
    }

    /// Start a new, empty effect [`Pipeline`] bound to this context.
    pub fn pipeline(&self) -> Pipeline<'_> {
        Pipeline::new(self)
    }

    pub(crate) fn as_ptr(&self) -> *mut fx_sys::fx_context_t {
        self.ptr
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context").finish_non_exhaustive()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: we own this handle and free it only here, exactly once.
        unsafe { fx_sys::fx_context_destroy(self.ptr) }
    }
}

/// The library version string, e.g. `"0.1.0"` (distinct from the `.so` soname).
pub fn version() -> &'static str {
    // SAFETY: `fx_version` returns a static, never-NULL C string.
    unsafe { CStr::from_ptr(fx_sys::fx_version()).to_str().unwrap_or("") }
}
