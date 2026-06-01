//! Rust mirror of the C `fx_cli`: load an image, apply a chain of effects
//! left-to-right, write the result. Works in RGBA8 throughout so the alpha
//! effects (shadow/rounded) are always legal.
//!
//! Run (needs the `image` feature, which the manifest requires for this example):
//!
//! ```text
//! cargo run -p fx --features image --example still_image -- \
//!     in.png out.png --gaussian 8 --gamma 0.9 [--gpu]
//! ```
//!
//! Effects (each consumes the values that follow it):
//!   --gamma <v>            --gaussian <sigma>     --kawase <offset>
//!   --color <b> <c> <s>    --rounded <radius> <softness>
//!   --shadow <sigma> <dx> <dy> <opacity>   --gpu

use std::error::Error;
use std::process::ExitCode;

use fx::{Backend, Context, Image, Pipeline, Shadow};

enum Op {
    Gamma(f32),
    Gaussian(f32),
    Kawase(f32),
    Color(f32, f32, f32),
    Rounded(f32, f32),
    Shadow(Shadow),
}

struct Args {
    input: String,
    output: String,
    backend: Backend,
    ops: Vec<Op>,
}

fn parse(mut it: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut backend = Backend::Cpu;
    let mut ops = Vec::new();

    // Pull the next token as an f32, or fail with which flag wanted it.
    let next_f32 = |it: &mut dyn Iterator<Item = String>, flag: &str| -> Result<f32, String> {
        it.next()
            .ok_or_else(|| format!("{flag} expects a number"))?
            .parse::<f32>()
            .map_err(|_| format!("{flag} expects a number"))
    };

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--gpu" => backend = Backend::Gpu,
            "--gamma" => ops.push(Op::Gamma(next_f32(&mut it, "--gamma")?)),
            "--gaussian" => ops.push(Op::Gaussian(next_f32(&mut it, "--gaussian")?)),
            "--kawase" => ops.push(Op::Kawase(next_f32(&mut it, "--kawase")?)),
            "--color" => ops.push(Op::Color(
                next_f32(&mut it, "--color")?,
                next_f32(&mut it, "--color")?,
                next_f32(&mut it, "--color")?,
            )),
            "--rounded" => ops.push(Op::Rounded(
                next_f32(&mut it, "--rounded")?,
                next_f32(&mut it, "--rounded")?,
            )),
            "--shadow" => ops.push(Op::Shadow(Shadow {
                sigma: next_f32(&mut it, "--shadow")?,
                dx: next_f32(&mut it, "--shadow")?,
                dy: next_f32(&mut it, "--shadow")?,
                opacity: next_f32(&mut it, "--shadow")?,
                ..Default::default()
            })),
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            _ => positional.push(arg),
        }
    }

    if positional.len() != 2 {
        return Err("usage: still_image <in> <out> [effects...] [--gpu]".into());
    }
    Ok(Args { input: positional.remove(0), output: positional.remove(0), backend, ops })
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse(std::env::args().skip(1))?;

    let ctx = Context::new(args.backend)?;
    let src = image::open(&args.input)?.to_rgba8();
    let input = Image::from_rgba8(&ctx, &src)?;
    let mut output = Image::new(&ctx, input.width(), input.height(), input.channels())?;

    let mut p: Pipeline = ctx.pipeline();
    for op in &args.ops {
        p = match *op {
            Op::Gamma(v) => p.gamma(v),
            Op::Gaussian(s) => p.gaussian(s),
            Op::Kawase(o) => p.kawase(o),
            Op::Color(b, c, s) => p.color(b, c, s),
            Op::Rounded(r, s) => p.rounded(r, s),
            Op::Shadow(s) => p.shadow(s),
        };
    }
    p.run(&input, &mut output)?;

    output.to_rgba8()?.save(&args.output)?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("still_image: {e}");
            ExitCode::FAILURE
        }
    }
}
