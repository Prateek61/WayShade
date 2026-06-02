// Exercises the safe wrapper end to end on the CPU backend: handle lifecycle,
// the data round-trip, the builder, and every error path. Needs the C side
// built, which comes in transitively via fx-sys.
use fx::{Backend, Context, FxError, Image, Shadow};

fn rgba(w: i32, h: i32) -> Vec<u8> {
    (0..(w * h * 4) as usize).map(|i| (i * 37 + 11) as u8).collect()
}

#[test]
fn version_is_reported() {
    assert!(!fx::version().is_empty());
}

#[test]
fn image_data_round_trips() {
    let ctx = Context::new(Backend::Cpu).unwrap();
    let pixels = rgba(16, 8);
    let img = Image::from_data(&ctx, 16, 8, 4, &pixels).unwrap();
    assert_eq!((img.width(), img.height(), img.channels()), (16, 8, 4));
    assert_eq!(img.data(), &pixels[..]);
}

#[test]
fn empty_pipeline_is_a_copy() {
    let ctx = Context::new(Backend::Cpu).unwrap();
    let pixels = rgba(32, 32);
    let input = Image::from_data(&ctx, 32, 32, 4, &pixels).unwrap();
    let mut output = Image::new(&ctx, 32, 32, 4).unwrap();
    ctx.pipeline().run(&input, &mut output).unwrap();
    assert_eq!(input.data(), output.data());
}

#[test]
fn gamma_one_is_identity() {
    let ctx = Context::new(Backend::Cpu).unwrap();
    let pixels = rgba(24, 24);
    let input = Image::from_data(&ctx, 24, 24, 4, &pixels).unwrap();
    let mut output = Image::new(&ctx, 24, 24, 4).unwrap();
    ctx.pipeline().gamma(1.0).run(&input, &mut output).unwrap();
    let max = input
        .data()
        .iter()
        .zip(output.data())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap();
    assert!(max <= 1, "gamma=1 should round-trip within 1, got {max}");
}

#[test]
fn multi_effect_chain_runs() {
    let ctx = Context::new(Backend::Cpu).unwrap();
    let pixels = rgba(40, 40);
    let input = Image::from_data(&ctx, 40, 40, 4, &pixels).unwrap();
    let mut output = Image::new(&ctx, 40, 40, 4).unwrap();
    ctx.pipeline()
        .gaussian(4.0)
        .shadow(Shadow { dy: 6.0, ..Default::default() })
        .gamma(0.9)
        .run(&input, &mut output)
        .unwrap();
}

#[test]
fn alpha_effect_on_rgb_is_unsupported() {
    let ctx = Context::new(Backend::Cpu).unwrap();
    let pixels = vec![128u8; 8 * 8 * 3];
    let input = Image::from_data(&ctx, 8, 8, 3, &pixels).unwrap();
    let mut output = Image::new(&ctx, 8, 8, 3).unwrap();
    let err = ctx.pipeline().rounded(4.0, 1.0).run(&input, &mut output).unwrap_err();
    assert!(matches!(err, FxError::Unsupported(_)), "got {err:?}");
}

#[test]
fn out_of_range_param_is_invalid_argument() {
    let ctx = Context::new(Backend::Cpu).unwrap();
    let pixels = rgba(8, 8);
    let input = Image::from_data(&ctx, 8, 8, 4, &pixels).unwrap();
    let mut output = Image::new(&ctx, 8, 8, 4).unwrap();
    let err = ctx.pipeline().gaussian(-1.0).run(&input, &mut output).unwrap_err();
    assert!(matches!(err, FxError::InvalidArgument(_)), "got {err:?}");
    // The detail string from the context should be populated.
    if let FxError::InvalidArgument(d) = &err {
        assert!(!d.is_empty(), "expected a detail message");
    }
}

#[test]
fn dimension_mismatch_is_caught() {
    let ctx = Context::new(Backend::Cpu).unwrap();
    let input = Image::new(&ctx, 8, 8, 4).unwrap();
    let mut output = Image::new(&ctx, 8, 9, 4).unwrap();
    let err = ctx.pipeline().gamma(1.0).run(&input, &mut output).unwrap_err();
    assert!(matches!(err, FxError::DimensionMismatch(_)), "got {err:?}");
}

#[test]
fn bad_from_data_length_is_rejected() {
    let ctx = Context::new(Backend::Cpu).unwrap();
    let err = Image::from_data(&ctx, 8, 8, 4, &[0u8; 10]).unwrap_err();
    assert!(matches!(err, FxError::InvalidArgument(_)), "got {err:?}");
}
