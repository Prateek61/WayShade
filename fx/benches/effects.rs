//! Per-frame benchmarks for every effect at 1080p / 1440p / 4K, on the CPU and
//! (when a CUDA driver is present) the GPU. One iteration is one `Pipeline::run`,
//! the compositor's per-frame cost. On the GPU that call also covers the upload,
//! kernel, copy-back, and device free, so it is real frame cost, not kernel time.
//!
//! Run length is tunable via env vars (defaults give a ~3-5 min full run):
//!   FX_BENCH_SAMPLE_SIZE   (default 10, criterion's floor)
//!   FX_BENCH_MEASURE_SECS  (default 3.0)
//!   FX_BENCH_WARMUP_SECS   (default 1.0)

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use fx::{Backend, Context, Image, Shadow};

const RESOLUTIONS: &[(i32, i32, &str)] = &[
    (1920, 1080, "1080p"),
    (2560, 1440, "1440p"),
    (3840, 2160, "4k"),
];

#[derive(Clone, Copy)]
enum Fx {
    Gamma,
    Gaussian,
    Kawase,
    Shadow,
    Color,
    Rounded,
}

const ALL: &[Fx] = &[Fx::Gamma, Fx::Gaussian, Fx::Kawase, Fx::Shadow, Fx::Color, Fx::Rounded];

impl Fx {
    fn name(self) -> &'static str {
        match self {
            Fx::Gamma => "gamma",
            Fx::Gaussian => "gaussian",
            Fx::Kawase => "kawase",
            Fx::Shadow => "shadow",
            Fx::Color => "color",
            Fx::Rounded => "rounded",
        }
    }

    // Only the blurs have a GPU schedule (see "GPU scope" in CLAUDE.md).
    fn has_gpu(self) -> bool {
        matches!(self, Fx::Gaussian | Fx::Kawase | Fx::Shadow)
    }

    // Same params BENCHMARKS.md quotes, so the table and the runs line up.
    fn build<'c>(self, ctx: &'c Context) -> fx::Pipeline<'c> {
        let p = ctx.pipeline();
        match self {
            Fx::Gamma => p.gamma(2.2),
            Fx::Gaussian => p.gaussian(8.0),
            Fx::Kawase => p.kawase(1.0),
            Fx::Shadow => p.shadow(Shadow::default()),
            Fx::Color => p.color(1.1, 1.2, 1.3),
            Fx::Rounded => p.rounded(16.0, 1.0),
        }
    }
}

// Deterministic RGBA fill. The kernels are data-independent, so content does
// not move timing, but a fixed pattern keeps runs comparable. 4-channel so the
// alpha effects (shadow/rounded) are legal at every size.
fn rgba_noise(w: i32, h: i32) -> Vec<u8> {
    let mut v = vec![0u8; w as usize * h as usize * 4];
    let mut s: u32 = 0x9e37_79b9;
    for b in v.iter_mut() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *b = (s >> 24) as u8;
    }
    v
}

// dlopen probe, same idea as tests/fx_capi_test.c. The first GPU op on a
// driverless box aborts the whole process (uncatchable cuInit), so gate the GPU
// groups on the driver actually loading instead of risking the bench binary.
fn cuda_available() -> bool {
    ["libcuda.so.1", "libcuda.so", "/usr/lib/wsl/lib/libcuda.so.1"]
        .iter()
        // SAFETY: we only load the driver to see if it's there. No symbols are
        // called and the handle drops right away.
        .any(|n| unsafe { libloading::Library::new(n).is_ok() })
}

// Build the pipeline once and reuse it across frames, then time `run` at each
// resolution with fresh input/output buffers.
fn bench_effect(group: &mut BenchmarkGroup<'_, WallTime>, ctx: &Context, fx: Fx, backend: &str) {
    let pipe = fx.build(ctx);
    for &(w, h, res) in RESOLUTIONS {
        let input = Image::from_data(ctx, w, h, 4, &rgba_noise(w, h)).expect("input image");
        let mut output = Image::new(ctx, w, h, 4).expect("output image");
        group.throughput(Throughput::Elements(w as u64 * h as u64));
        group.bench_function(format!("{backend}/{res}"), |b| {
            b.iter(|| pipe.run(&input, &mut output).expect("run"));
        });
    }
}

fn bench_all(c: &mut Criterion) {
    let cpu = Context::new(Backend::Cpu).expect("cpu context");
    let gpu = if cuda_available() {
        Some(Context::new(Backend::Gpu).expect("gpu context"))
    } else {
        eprintln!("skip: no CUDA driver found, GPU benchmarks skipped");
        None
    };

    for &fx in ALL {
        let mut group = c.benchmark_group(fx.name());
        bench_effect(&mut group, &cpu, fx, "cpu");
        if fx.has_gpu()
            && let Some(gpu) = gpu.as_ref()
        {
            bench_effect(&mut group, gpu, fx, "gpu");
        }
        group.finish();
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn configured() -> Criterion {
    // criterion's sample_size floor is 10, so clamp up to avoid a panic.
    let sample = env_usize("FX_BENCH_SAMPLE_SIZE", 10).max(10);
    Criterion::default()
        .sample_size(sample)
        .measurement_time(Duration::from_secs_f64(env_f64("FX_BENCH_MEASURE_SECS", 3.0)))
        .warm_up_time(Duration::from_secs_f64(env_f64("FX_BENCH_WARMUP_SECS", 1.0)))
}

criterion_group! {
    name = benches;
    config = configured();
    targets = bench_all
}
criterion_main!(benches);
