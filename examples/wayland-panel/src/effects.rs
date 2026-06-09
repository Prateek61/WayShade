// Live backdrop blur over libfx. Holds the context for the whole run; the image and
// pipeline handles are built per call so their `'ctx` borrow never has to outlive a
// frame (a struct holding both the context and handles borrowing it is self-referential).

use fx::{Backend, Context, FxError, Image};

pub struct Blur {
    ctx: Context,
    offset: f32, // dual-Kawase strength (radius grows with it); the runtime intensity knob
}

impl Blur {
    pub fn new(gpu: bool, offset: f32) -> Result<Self, FxError> {
        let backend = if gpu { Backend::Gpu } else { Backend::Cpu };
        Ok(Blur { ctx: Context::new(backend)?, offset })
    }

    // Blur a w*h BGRA strip in place. The blur is per-channel and linear, so feeding
    // B,G,R,A gives the same result as R,G,B,A swapped back, no reorder needed; the
    // alpha is a constant 255 and edge-replicate keeps it 255.
    pub fn run(&self, buf: &mut [u8], w: u32, h: u32) -> Result<(), FxError> {
        let input = Image::from_data(&self.ctx, w as i32, h as i32, 4, buf)?;
        let mut output = Image::new(&self.ctx, w as i32, h as i32, 4)?;
        self.ctx.pipeline().kawase(self.offset).run(&input, &mut output)?;
        buf.copy_from_slice(output.data());
        Ok(())
    }
}

// Blend a tint over the (already blurred) BGRA strip, in place; the panel stays
// opaque so premultiplied ARGB equals straight. `alpha` is the tint strength: 0
// leaves the backdrop untouched, 255 paints the flat tint color.
pub fn composite_tint(canvas: &mut [u8], tint: [u8; 3], alpha: u8) {
    let [r, g, b] = tint;
    let t = alpha as u16;
    let mix = |bg: u8, fg: u8| ((bg as u16 * (255 - t) + fg as u16 * t) / 255) as u8;
    for px in canvas.chunks_exact_mut(4) {
        px[0] = mix(px[0], b);
        px[1] = mix(px[1], g);
        px[2] = mix(px[2], r);
        px[3] = 255;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tint_alpha_zero_is_passthrough() {
        let mut c = vec![10, 20, 30, 255];
        composite_tint(&mut c, [200, 100, 50], 0);
        assert_eq!(c, vec![10, 20, 30, 255]);
    }

    #[test]
    fn tint_alpha_full_paints_color() {
        // tint R200 G100 B50 lands in panel byte order B=50, G=100, R=200.
        let mut c = vec![10, 20, 30, 0];
        composite_tint(&mut c, [200, 100, 50], 255);
        assert_eq!(c, vec![50, 100, 200, 255]);
    }
}
