// The backdrop the panel mirrors. Two sources, picked at runtime: the real one
// drives wlr-screencopy (capture the output behind us each frame); the synthetic
// one paints an animated test pattern so the blit/timing plumbing is exercisable
// on compositors that don't expose screencopy.

use std::time::{Duration, Instant};

use smithay_client_toolkit::shm::{Shm, raw::RawPool};
use wayland_client::{
    Proxy, QueueHandle, WEnum,
    protocol::{wl_buffer::WlBuffer, wl_output::WlOutput, wl_shm},
};
use wayland_protocols_wlr::screencopy::v1::client::{
    zwlr_screencopy_frame_v1::ZwlrScreencopyFrameV1,
    zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
};

use crate::app::App;

// A captured image in the panel's own layout: ARGB8888 premultiplied, little-endian
// bytes B,G,R,A, so a row is a memcpy straight into the surface canvas.
#[derive(Default)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Frame {
    // Copy a (sx,sy)-anchored rectangle of this frame into dst (a dw*dh BGRA canvas).
    // Returns false when there's nothing captured yet so the caller can fall back.
    fn blit(&self, dst: &mut [u8], dw: u32, dh: u32, sx: u32, sy: u32) -> bool {
        if self.pixels.is_empty() || self.width == 0 {
            return false;
        }
        let cols = dw.min(self.width.saturating_sub(sx)) as usize;
        if cols == 0 {
            return false;
        }
        for y in 0..dh {
            let syy = (sy + y).min(self.height - 1);
            let so = ((syy * self.width + sx) * 4) as usize;
            let do_ = (y * dw * 4) as usize;
            let n = cols * 4;
            dst[do_..do_ + n].copy_from_slice(&self.pixels[so..so + n]);
        }
        true
    }
}

pub enum Backdrop {
    Screencopy(Box<Capture>), // boxed: Capture is an order of magnitude larger than Synthetic
    Synthetic(Synthetic),
}

impl Backdrop {
    // Fill the panel canvas from the latest capture. Real source mirrors the strip
    // just outside the bar (feedback-free: the bar never covers those rows);
    // synthetic source is already panel-sized so it blits 1:1.
    pub fn present(&self, dst: &mut [u8], w: u32, h: u32) -> bool {
        match self {
            Backdrop::Synthetic(s) => s.frame.blit(dst, w, h, 0, 0),
            Backdrop::Screencopy(c) => {
                let sy = if c.anchor_top {
                    h
                } else {
                    c.latest.height.saturating_sub(2 * h)
                };
                c.latest.blit(dst, w, h, 0, sy)
            }
        }
    }
}

// ---- synthetic source -------------------------------------------------------

#[derive(Default)]
pub struct Synthetic {
    frame: Frame,
}

impl Synthetic {
    pub fn new() -> Self {
        Synthetic::default()
    }

    // A scrolling diagonal gradient keyed off the frame-callback time, so the panel
    // visibly animates and we can confirm the per-frame blit path actually runs.
    pub fn fill(&mut self, time: u32, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        let f = &mut self.frame;
        f.pixels.resize((w * h * 4) as usize, 0);
        f.width = w;
        f.height = h;
        let phase = time / 8;
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                f.pixels[i] = (x.wrapping_add(y).wrapping_add(phase) / 2) as u8; // B
                f.pixels[i + 1] = y.wrapping_add(phase) as u8; // G
                f.pixels[i + 2] = x.wrapping_add(phase) as u8; // R
                f.pixels[i + 3] = 255;
            }
        }
    }
}

// ---- screencopy source ------------------------------------------------------

// Whole-output capture via wlr-screencopy. One capture in flight at a time and one
// reused shm buffer: `ready` means the copy is done, so we can read it and hand the
// same buffer to the next copy without tracking wl_buffer.release.
pub struct Capture {
    manager: ZwlrScreencopyManagerV1,
    cursor: bool,
    anchor_top: bool,
    pool: RawPool,
    buffer: Option<WlBuffer>,
    buf_dims: Option<(u32, u32, u32, wl_shm::Format)>, // w, h, stride, format
    frame: Option<ZwlrScreencopyFrameV1>,              // in-flight, else idle
    pending: Option<(wl_shm::Format, u32, u32, u32)>,  // from the shm `buffer` event
    y_invert: bool,
    latest: Frame,
    requested_at: Option<Instant>,
    stat_sum: Duration,
    stat_max: Duration,
    stat_count: u32,
}

// A 1920x1200 BGRA frame; resize() grows it if the real output is larger.
const DEFAULT_POOL: usize = 1920 * 1200 * 4;

impl Capture {
    pub fn new(manager: ZwlrScreencopyManagerV1, shm: &Shm, cursor: bool, anchor_top: bool) -> Self {
        let pool = RawPool::new(DEFAULT_POOL, shm).expect("create capture shm pool");
        Capture {
            manager,
            cursor,
            anchor_top,
            pool,
            buffer: None,
            buf_dims: None,
            frame: None,
            pending: None,
            y_invert: false,
            latest: Frame::default(),
            requested_at: None,
            stat_sum: Duration::ZERO,
            stat_max: Duration::ZERO,
            stat_count: 0,
        }
    }

    // Kick off a capture for `output`, unless one's already in flight (we pipeline
    // one frame behind present rather than blocking the frame callback on `ready`).
    pub fn begin(&mut self, output: &WlOutput, qh: &QueueHandle<App>) {
        if self.frame.is_some() {
            return;
        }
        self.pending = None;
        self.y_invert = false;
        self.requested_at = Some(Instant::now());
        self.frame = Some(self.manager.capture_output(self.cursor as i32, output, qh, ()));
    }

    pub fn on_buffer(&mut self, format: WEnum<wl_shm::Format>, w: u32, h: u32, stride: u32, qh: &QueueHandle<App>) {
        if let WEnum::Value(fmt) = format {
            self.pending = Some((fmt, w, h, stride));
        }
        // v1/v2 managers send no buffer_done, so the first shm `buffer` is our cue.
        if self.manager.version() < 3 {
            self.copy(qh);
        }
    }

    pub fn on_buffer_done(&mut self, qh: &QueueHandle<App>) {
        self.copy(qh);
    }

    pub fn set_y_invert(&mut self, inv: bool) {
        self.y_invert = inv;
    }

    fn copy(&mut self, qh: &QueueHandle<App>) {
        let (Some(frame), Some((fmt, w, h, stride))) = (self.frame.as_ref(), self.pending) else {
            return;
        };
        let _ = self.pool.resize((stride * h) as usize);
        if self.buf_dims != Some((w, h, stride, fmt)) {
            if let Some(b) = self.buffer.take() {
                b.destroy();
            }
            self.buffer = Some(self.pool.create_buffer(0, w as i32, h as i32, stride as i32, fmt, (), qh));
            self.buf_dims = Some((w, h, stride, fmt));
        }
        frame.copy(self.buffer.as_ref().unwrap());
    }

    // Copy finished: read the shm buffer into `latest` (panel layout) and record the
    // request->ready latency, then destroy the frame and go idle for the next begin().
    pub fn on_ready(&mut self) {
        if let (Some((w, h, stride, fmt)), Some(t0)) = (self.buf_dims, self.requested_at) {
            self.latest.pixels.resize((w * h * 4) as usize, 0);
            self.latest.width = w;
            self.latest.height = h;
            let src = self.pool.mmap();
            for r in 0..h {
                let so = (r * stride) as usize;
                let dr = if self.y_invert { h - 1 - r } else { r };
                let do_ = (dr * w * 4) as usize;
                convert_row(&src[so..so + (w * 4) as usize], &mut self.latest.pixels[do_..do_ + (w * 4) as usize], fmt);
            }
            let dt = t0.elapsed();
            self.stat_sum += dt;
            self.stat_max = self.stat_max.max(dt);
            self.stat_count += 1;
        }
        self.finish();
    }

    pub fn on_failed(&mut self) {
        self.finish();
    }

    fn finish(&mut self) {
        if let Some(f) = self.frame.take() {
            f.destroy();
        }
        self.requested_at = None;
    }

    // Drain the accumulated capture-latency stats for the periodic report.
    pub fn drain_stats(&mut self) -> Option<(f32, f32, u32)> {
        if self.stat_count == 0 {
            return None;
        }
        let avg = self.stat_sum.as_secs_f32() * 1000.0 / self.stat_count as f32;
        let max = self.stat_max.as_secs_f32() * 1000.0;
        let n = self.stat_count;
        self.stat_sum = Duration::ZERO;
        self.stat_max = Duration::ZERO;
        self.stat_count = 0;
        Some((avg, max, n))
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if let Some(f) = self.frame.take() {
            f.destroy();
        }
        if let Some(b) = self.buffer.take() {
            b.destroy();
        }
        self.manager.destroy();
    }
}

// One source row -> one panel row (BGRA, forced opaque). wlroots hands outputs back
// as XRGB/ARGB (already B,G,R,_ little-endian) or the BGR variants (R and B swapped).
fn convert_row(src: &[u8], dst: &mut [u8], fmt: wl_shm::Format) {
    use wl_shm::Format::{Abgr8888, Xbgr8888};
    let swap = matches!(fmt, Xbgr8888 | Abgr8888);
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        if swap {
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
        } else {
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        }
        d[3] = 255;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_xrgb_passthrough_opaque() {
        // XRGB/ARGB are already B,G,R,_ in shm byte order copy through, force opaque.
        let src = [10, 20, 30, 0];
        let mut dst = [0u8; 4];
        convert_row(&src, &mut dst, wl_shm::Format::Xrgb8888);
        assert_eq!(dst, [10, 20, 30, 255]);
    }

    #[test]
    fn convert_xbgr_swaps_r_and_b() {
        // XBGR is R,G,B,_ , R and B must swap to land in panel B,G,R order.
        let src = [10, 20, 30, 0]; // R=10 G=20 B=30
        let mut dst = [0u8; 4];
        convert_row(&src, &mut dst, wl_shm::Format::Xbgr8888);
        assert_eq!(dst, [30, 20, 10, 255]); // B=30 G=20 R=10
    }

    #[test]
    fn blit_offsets_and_clamps() {
        // 2x3 source; blit a 2x1 panel anchored at row 1 (the "strip below the bar").
        let mut src = Frame { width: 2, height: 3, pixels: vec![0; 2 * 3 * 4] };
        for p in 0..6 {
            src.pixels[p * 4] = p as u8; // tag each pixel by its B channel
        }
        let mut dst = vec![0u8; 2 * 4];
        assert!(src.blit(&mut dst, 2, 1, 0, 1)); // row 1 = source pixels 2,3
        assert_eq!(dst[0], 2);
        assert_eq!(dst[4], 3);
    }

    #[test]
    fn blit_empty_frame_reports_unfilled() {
        let empty = Frame::default();
        let mut dst = vec![0u8; 16];
        assert!(!empty.blit(&mut dst, 2, 2, 0, 0));
    }

    #[test]
    fn synthetic_fills_opaque_panel_sized() {
        let mut s = Synthetic::new();
        s.fill(0, 4, 2);
        assert_eq!(s.frame.width, 4);
        assert_eq!(s.frame.height, 2);
        assert_eq!(s.frame.pixels.len(), 4 * 2 * 4);
        assert!(s.frame.pixels.chunks_exact(4).all(|px| px[3] == 255));
    }
}
