// Locate-only, mirroring fx-sys: re-emit an rpath so this crate's test, example,
// and doctest binaries find libfx.so.1 with no LD_LIBRARY_PATH. fx-sys already
// emits the link-search/link-lib directives transitively; only the rpath needs
// to land on *our* binaries.
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=FX_LIB_DIR");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest.parent().unwrap();
    let lib_dir = env::var("FX_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace.join("build/src"));

    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // WSL ships the CUDA driver here; bake it in so the `--gpu` example works
    // without LD_LIBRARY_PATH. No-op off WSL (Stage 1 gotcha #10).
    if Path::new("/usr/lib/wsl/lib").exists() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/wsl/lib");
    }
}
