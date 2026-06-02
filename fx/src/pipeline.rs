use std::ptr;

use fx_sys::{fx_pipeline_t, fx_status_t};

use crate::error::last_error_string;
use crate::{Context, FxError, Image, Result};

/// Drop-shadow parameters. Construct with [`Default`] and override fields:
///
/// ```
/// use fx::Shadow;
/// let s = Shadow { sigma: 16.0, dy: 24.0, opacity: 0.8, ..Default::default() };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shadow {
    /// Blur strength of the shadow's alpha (> 0).
    pub sigma: f32,
    /// Horizontal offset in pixels.
    pub dx: f32,
    /// Vertical offset in pixels.
    pub dy: f32,
    /// Shadow tint `[r, g, b]`.
    pub tint: [u8; 3],
    /// Shadow strength in `[0, 1]`.
    pub opacity: f32,
}

impl Default for Shadow {
    /// Matches the C ABI / CLI defaults: `sigma 8, dx 0, dy 8, black, opacity 0.5`.
    fn default() -> Self {
        Shadow { sigma: 8.0, dx: 0.0, dy: 8.0, tint: [0, 0, 0], opacity: 0.5 }
    }
}

/// An ordered chain of effects, applied left-to-right by [`Pipeline::run`].
///
/// Built fluently from [`Context::pipeline`]. Each effect method validates its
/// parameters immediately; the first failure is stashed and surfaced by `run`,
/// so a whole chain needs only one `?`:
///
/// ```no_run
/// # use fx::{Backend, Context, Image};
/// # fn main() -> Result<(), fx::FxError> {
/// let ctx = Context::new(Backend::Cpu)?;
/// let input = Image::new(&ctx, 64, 64, 4)?;
/// let mut output = Image::new(&ctx, 64, 64, 4)?;
/// ctx.pipeline()
///     .kawase(1.0)
///     .gamma(0.9)
///     .run(&input, &mut output)?;
/// # Ok(()) }
/// ```
///
/// A built pipeline can be re-run on many images (`run` borrows `&self`), which
/// is the per-frame compositor case.
pub struct Pipeline<'ctx> {
    ptr: *mut fx_pipeline_t,
    ctx: &'ctx Context,
    err: Option<FxError>,
}

impl<'ctx> Pipeline<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context) -> Self {
        let mut ptr: *mut fx_pipeline_t = ptr::null_mut();
        // SAFETY: `ctx` is live; `out_pipe` is a valid out-pointer.
        let status = unsafe { fx_sys::fx_pipeline_create(ctx.as_ptr(), &mut ptr) };
        let err = (status != fx_status_t::FX_OK)
            .then(|| FxError::from_status(status, last_error_string(ctx.as_ptr())));
        Pipeline { ptr, ctx, err }
    }

    /// Append one effect via `f`, keeping only the first failure. Once an earlier
    /// step has errored this does nothing, so `ptr` is never touched after a
    /// failed create.
    fn step<F>(mut self, f: F) -> Self
    where
        F: FnOnce(*mut fx_pipeline_t) -> fx_status_t,
    {
        if self.err.is_none() {
            let status = f(self.ptr);
            if status != fx_status_t::FX_OK {
                self.err = Some(FxError::from_status(status, last_error_string(self.ctx.as_ptr())));
            }
        }
        self
    }

    /// Gamma correction. `value > 0`; `> 1` brightens midtones, `< 1` darkens.
    pub fn gamma(self, value: f32) -> Self {
        self.step(|p| unsafe { fx_sys::fx_pipeline_gamma(p, value) })
    }

    /// Separable Gaussian blur. `sigma > 0`.
    pub fn gaussian(self, sigma: f32) -> Self {
        self.step(|p| unsafe { fx_sys::fx_pipeline_gaussian(p, sigma) })
    }

    /// Dual-Kawase blur. `offset > 0` scales the tap distance (blur strength).
    pub fn kawase(self, offset: f32) -> Self {
        self.step(|p| unsafe { fx_sys::fx_pipeline_kawase(p, offset) })
    }

    /// Color correction. Each factor `>= 0`, `1.0` is identity.
    pub fn color(self, brightness: f32, contrast: f32, saturation: f32) -> Self {
        self.step(|p| unsafe { fx_sys::fx_pipeline_color(p, brightness, contrast, saturation) })
    }

    /// Rounded-corner alpha mask. Needs a 4-channel image at run time.
    pub fn rounded(self, radius: f32, softness: f32) -> Self {
        self.step(|p| unsafe { fx_sys::fx_pipeline_rounded(p, radius, softness) })
    }

    /// Drop shadow. Needs a 4-channel image at run time. See [`Shadow`].
    pub fn shadow(self, s: Shadow) -> Self {
        self.step(|p| unsafe {
            fx_sys::fx_pipeline_shadow(p, s.sigma, s.dx, s.dy, s.tint[0], s.tint[1], s.tint[2], s.opacity)
        })
    }

    /// Apply the chain into `output`, leaving `input` untouched. `output` must
    /// match `input` in width, height, and channels. If any effect needs alpha,
    /// both must be 4-channel or this returns [`FxError::Unsupported`].
    pub fn run(&self, input: &Image<'_>, output: &mut Image<'_>) -> Result<()> {
        if let Some(e) = &self.err {
            return Err(e.clone());
        }
        if (input.width(), input.height(), input.channels())
            != (output.width(), output.height(), output.channels())
        {
            return Err(FxError::DimensionMismatch(format!(
                "input {}x{}x{} != output {}x{}x{}",
                input.width(),
                input.height(),
                input.channels(),
                output.width(),
                output.height(),
                output.channels()
            )));
        }
        // SAFETY: err is None, so create succeeded and self.ptr is a live
        // pipeline; the image handles are live and their dimensions match.
        let status = unsafe { fx_sys::fx_pipeline_run(self.ptr, input.as_ptr(), output.as_mut_ptr()) };
        if status != fx_status_t::FX_OK {
            return Err(FxError::from_status(status, last_error_string(self.ctx.as_ptr())));
        }
        Ok(())
    }
}

impl std::fmt::Debug for Pipeline<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline").field("error", &self.err).finish()
    }
}

impl Drop for Pipeline<'_> {
    fn drop(&mut self) {
        // SAFETY: we own this handle. It is NULL only if create failed, and
        // fx_pipeline_destroy tolerates NULL.
        unsafe { fx_sys::fx_pipeline_destroy(self.ptr) }
    }
}
