// Locate-only, mirroring fx-sys. We re-emit an rpath so this crate's test,
// example, and doctest binaries find libfx.so.1 with no LD_LIBRARY_PATH. The
// link-search/link-lib directives already arrive transitively from fx-sys, so
// only the rpath has to land on our own binaries.
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

    // WSL keeps the CUDA driver here, so bake it in for the `--gpu` example to
    // work without LD_LIBRARY_PATH. No-op off WSL.
    if Path::new("/usr/lib/wsl/lib").exists() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/wsl/lib");
    }
}
