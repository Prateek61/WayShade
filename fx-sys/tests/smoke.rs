use std::ffi::CStr;
use std::ptr;

// The unsafe layer: create and destroy a context, the minimal proof the
// bindings link and the ABI round-trips.
#[test]
fn context_create_destroy() {
    unsafe {
        let version = CStr::from_ptr(fx_sys::fx_version());
        println!("libfx version: {}", version.to_str().unwrap());

        let mut ctx: *mut fx_sys::fx_context_t = ptr::null_mut();
        let status = fx_sys::fx_context_create(fx_sys::fx_backend_t::FX_BACKEND_CPU, &mut ctx);
        assert_eq!(status, fx_sys::fx_status_t::FX_OK);
        assert!(!ctx.is_null());

        fx_sys::fx_context_destroy(ctx);
    }
}
