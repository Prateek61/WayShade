use std::ffi::CStr;
use std::fmt;

use fx_sys::{fx_context_last_error, fx_context_t, fx_status_t};

/// An error returned by a `libfx` call.
///
/// Each variant maps to an `fx_status_t` code and carries the context's
/// last-error detail string.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FxError {
    /// A NULL handle, bad dimensions, or an out-of-range effect parameter.
    InvalidArgument(String),
    /// An allocation failed.
    OutOfMemory(String),
    /// The operation is not supported, e.g. an alpha effect on a 3-channel image.
    Unsupported(String),
    /// The run output does not match the input in width, height, or channels.
    DimensionMismatch(String),
    /// A Halide kernel or GPU device call failed.
    Backend(String),
    /// An unexpected internal failure.
    Internal(String),
    /// A status code the bindings do not model. Holds the raw code and detail.
    Unknown(i32, String),
}

impl FxError {
    /// Map a non-OK status plus a detail string to the matching variant.
    pub(crate) fn from_status(status: fx_status_t, detail: String) -> Self {
        match status {
            fx_status_t::FX_ERR_INVALID_ARGUMENT => FxError::InvalidArgument(detail),
            fx_status_t::FX_ERR_OUT_OF_MEMORY => FxError::OutOfMemory(detail),
            fx_status_t::FX_ERR_UNSUPPORTED => FxError::Unsupported(detail),
            fx_status_t::FX_ERR_DIMENSION_MISMATCH => FxError::DimensionMismatch(detail),
            fx_status_t::FX_ERR_BACKEND => FxError::Backend(detail),
            fx_status_t::FX_ERR_INTERNAL => FxError::Internal(detail),
            other => FxError::Unknown(other.0 as i32, detail),
        }
    }

    fn kind_str(&self) -> &'static str {
        match self {
            FxError::InvalidArgument(_) => "invalid argument",
            FxError::OutOfMemory(_) => "out of memory",
            FxError::Unsupported(_) => "unsupported",
            FxError::DimensionMismatch(_) => "dimension mismatch",
            FxError::Backend(_) => "backend error",
            FxError::Internal(_) => "internal error",
            FxError::Unknown(..) => "unknown error",
        }
    }

    fn detail(&self) -> &str {
        match self {
            FxError::InvalidArgument(d)
            | FxError::OutOfMemory(d)
            | FxError::Unsupported(d)
            | FxError::DimensionMismatch(d)
            | FxError::Backend(d)
            | FxError::Internal(d)
            | FxError::Unknown(_, d) => d,
        }
    }
}

impl fmt::Display for FxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = self.detail();
        if detail.is_empty() {
            write!(f, "{}", self.kind_str())
        } else {
            write!(f, "{}: {}", self.kind_str(), detail)
        }
    }
}

impl std::error::Error for FxError {}

/// Copy a context's last-error message out as an owned `String`. The C pointer
/// is only good until the next call, so copy it now. Empty if the context is
/// NULL or has no error to report.
pub(crate) fn last_error_string(ctx: *const fx_context_t) -> String {
    if ctx.is_null() {
        return String::new();
    }
    // SAFETY: ctx is live. The returned C string (or NULL) is owned by the
    // context, so copy it before handing back.
    unsafe {
        let p = fx_context_last_error(ctx);
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}
