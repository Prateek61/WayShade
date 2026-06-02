use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest_dir.parent().unwrap().to_path_buf();
    let include_dir = workspace.join("include");
    let header = include_dir.join("fx/fx.h");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-env-changed=FX_LIB_DIR");

    // Raw bindings from the public C ABI. Allowlist our own surface so transitive
    // stddef.h junk stays out. Newtype enums keep the raw layer sound if a kernel
    // ever returns a code we don't model, and the safe crate maps them later.
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("fx_.*")
        .allowlist_type("fx_.*")
        .allowlist_var("FX_.*")
        .newtype_enum("fx_status_t")
        .newtype_enum("fx_backend_t")
        .generate()
        .expect("failed to generate libfx bindings");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    // Locate-only: FX_LIB_DIR (the env var the C test suite already uses), else
    // the conventional build/src. No cmake fallback by design.
    let lib_dir = env::var("FX_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace.join("build/src"));

    if !lib_dir.join("libfx.so").exists() {
        panic!(
            "libfx not found in {}. Build the C side first \
             (cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build), \
             or set FX_LIB_DIR to the directory containing libfx.so.",
            lib_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=fx");
    // rpath so the test binary finds libfx.so.1 with no LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
