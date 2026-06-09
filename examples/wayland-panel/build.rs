// We link libfx through the `fx` crate, but `cargo:rustc-link-arg` (the rpath fx's
// own build.rs emits) doesn't propagate to dependents, so re-emit it here for this
// binary.
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=FX_LIB_DIR");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // examples/wayland-panel -> workspace root is two levels up.
    let workspace = manifest.parent().unwrap().parent().unwrap();
    let lib_dir = env::var("FX_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace.join("build/src"));

    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // WSL keeps the CUDA driver here; bake it in so --gpu works without
    // LD_LIBRARY_PATH. No-op off WSL.
    if Path::new("/usr/lib/wsl/lib").exists() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/wsl/lib");
    }
}
