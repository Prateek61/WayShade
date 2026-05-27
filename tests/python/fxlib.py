"""ctypes binding for libfx, used by the reference-parity tests.
"""

import ctypes
import os

import numpy as np

FX_OK = 0
FX_BACKEND_CPU = 0
FX_BACKEND_GPU = 1
FX_BACKEND_AUTO = 2

c_void_pp = ctypes.POINTER(ctypes.c_void_p)
c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)


def _find_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = []
    env = os.environ.get("FX_LIB_DIR")
    if env:
        candidates += [os.path.join(env, "libfx.so"), os.path.join(env, "libfx.so.1")]
    candidates += [
        os.path.join(repo, "build", "src", "libfx.so"),
        os.path.join(repo, "build", "src", "libfx.so.1"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "libfx.so not found. Build it first (cmake --build build) or set FX_LIB_DIR. "
        "Looked in:\n  " + "\n  ".join(candidates))


_lib = ctypes.CDLL(_find_lib())

_lib.fx_version.restype = ctypes.c_char_p
_lib.fx_status_string.argtypes = [ctypes.c_int]
_lib.fx_status_string.restype = ctypes.c_char_p
_lib.fx_context_create.argtypes = [ctypes.c_int, c_void_pp]
_lib.fx_context_create.restype = ctypes.c_int
_lib.fx_context_destroy.argtypes = [ctypes.c_void_p]
_lib.fx_context_last_error.argtypes = [ctypes.c_void_p]
_lib.fx_context_last_error.restype = ctypes.c_char_p
_lib.fx_image_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, c_void_pp]
_lib.fx_image_create.restype = ctypes.c_int
_lib.fx_image_from_data.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, c_ubyte_p, c_void_pp]
_lib.fx_image_from_data.restype = ctypes.c_int
_lib.fx_image_data.argtypes = [ctypes.c_void_p]
_lib.fx_image_data.restype = c_ubyte_p
_lib.fx_image_destroy.argtypes = [ctypes.c_void_p]
_lib.fx_pipeline_create.argtypes = [ctypes.c_void_p, c_void_pp]
_lib.fx_pipeline_create.restype = ctypes.c_int
_lib.fx_pipeline_gamma.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.fx_pipeline_gamma.restype = ctypes.c_int
_lib.fx_pipeline_gaussian.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.fx_pipeline_gaussian.restype = ctypes.c_int
_lib.fx_pipeline_kawase.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.fx_pipeline_kawase.restype = ctypes.c_int
_lib.fx_pipeline_color.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
_lib.fx_pipeline_color.restype = ctypes.c_int
_lib.fx_pipeline_rounded.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
_lib.fx_pipeline_rounded.restype = ctypes.c_int
_lib.fx_pipeline_shadow.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                    ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_float]
_lib.fx_pipeline_shadow.restype = ctypes.c_int
_lib.fx_pipeline_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.fx_pipeline_run.restype = ctypes.c_int
_lib.fx_pipeline_destroy.argtypes = [ctypes.c_void_p]


class FxError(RuntimeError):
    pass


def version():
    return _lib.fx_version().decode()


def status_string(status):
    return _lib.fx_status_string(status).decode()


# name -> (arity, append fn). The append fn forwards positional float/int args.
_APPENDERS = {
    "gamma":    lambda p, a: _lib.fx_pipeline_gamma(p, a[0]),
    "gaussian": lambda p, a: _lib.fx_pipeline_gaussian(p, a[0]),
    "kawase":   lambda p, a: _lib.fx_pipeline_kawase(p, a[0]),
    "color":    lambda p, a: _lib.fx_pipeline_color(p, a[0], a[1], a[2]),
    "rounded":  lambda p, a: _lib.fx_pipeline_rounded(p, a[0], a[1]),
    "shadow":   lambda p, a: _lib.fx_pipeline_shadow(p, a[0], a[1], a[2],
                                                     int(a[3]), int(a[4]), int(a[5]), a[6]),
}


def apply(arr, effects, backend=FX_BACKEND_CPU):
    """Run an effect chain on `arr` (H,W,C uint8) and return the result as a fresh
    H,W,C uint8 array. `effects` is a list of (name, args) tuples, e.g.
    [("gaussian", (4.0,)), ("color", (1.1, 1.0, 1.0))]. Raises FxError on failure."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError("expected an (H, W, C) array")
    h, w, c = arr.shape

    ctx = ctypes.c_void_p()
    if _lib.fx_context_create(backend, ctypes.byref(ctx)) != FX_OK:
        raise FxError("fx_context_create failed")

    img_in = ctypes.c_void_p()
    img_out = ctypes.c_void_p()
    pipe = ctypes.c_void_p()
    try:
        src = arr.ctypes.data_as(c_ubyte_p)
        _check(_lib.fx_image_from_data(ctx, w, h, c, src, ctypes.byref(img_in)), ctx)
        _check(_lib.fx_image_create(ctx, w, h, c, ctypes.byref(img_out)), ctx)
        _check(_lib.fx_pipeline_create(ctx, ctypes.byref(pipe)), ctx)

        for name, args in effects:
            if name not in _APPENDERS:
                raise ValueError("unknown effect: " + name)
            _check(_APPENDERS[name](pipe, args), ctx)

        _check(_lib.fx_pipeline_run(pipe, img_in, img_out), ctx)

        data = _lib.fx_image_data(img_out)
        out = np.ctypeslib.as_array(data, shape=(h, w, c)).copy()
        return out
    finally:
        if pipe:
            _lib.fx_pipeline_destroy(pipe)
        if img_out:
            _lib.fx_image_destroy(img_out)
        if img_in:
            _lib.fx_image_destroy(img_in)
        _lib.fx_context_destroy(ctx)


def _check(status, ctx):
    if status != FX_OK:
        detail = _lib.fx_context_last_error(ctx).decode()
        msg = status_string(status)
        raise FxError(msg + (": " + detail if detail else "") + " (status %d)" % status)
