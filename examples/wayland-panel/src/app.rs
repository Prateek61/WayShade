use std::time::{Duration, Instant};

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, delegate_shm,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        WaylandSurface,
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
    },
    shm::{Shm, ShmHandler, slot::SlotPool},
};
use wayland_client::{
    Connection, Dispatch, QueueHandle, WEnum,
    globals::GlobalList,
    protocol::{wl_buffer, wl_output, wl_shm, wl_surface},
};
use wayland_protocols_wlr::screencopy::v1::client::{
    zwlr_screencopy_frame_v1::{self, ZwlrScreencopyFrameV1},
    zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
};

use crate::backdrop::{Backdrop, Capture, Synthetic};
use crate::effect::{Blur, composite_tint};
use crate::{Config, Edge};

pub struct App {
    registry_state: RegistryState,
    output_state: OutputState,
    shm: Shm,
    pool: SlotPool,
    layer: LayerSurface,
    backdrop: Backdrop,
    blur: Option<Blur>,
    cfg: Config,
    width: u32,
    height: u32,
    time: u32,
    configured: bool,
    pub exit: bool,
    // The last raw captured strip and its blurred+tinted result. If a new capture
    // matches prev_raw the backdrop is unchanged, so we reuse cached and skip the blur.
    prev_raw: Vec<u8>,
    cached: Vec<u8>,
    // rough fps + capture-latency + blur-time accounting, reported once a second
    frames: u32,
    last_report: u32,
    blur_sum: Duration,
    blur_max: Duration,
    blur_count: u32,
    blur_skipped: u32,
}

impl App {
    pub fn new(globals: &GlobalList, qh: &QueueHandle<App>, cfg: Config) -> Self {
        let compositor = CompositorState::bind(globals, qh).expect("wl_compositor not available");
        let layer_shell = LayerShell::bind(globals, qh).expect("zwlr_layer_shell_v1 not available");
        let shm = Shm::bind(globals, qh).expect("wl_shm not available");

        let surface = compositor.create_surface(qh);
        let layer =
            layer_shell.create_layer_surface(qh, surface, Layer::Top, Some("wayshade-panel"), None);

        let edge = match cfg.anchor {
            Edge::Top => Anchor::TOP,
            Edge::Bottom => Anchor::BOTTOM,
        };
        layer.set_anchor(edge | Anchor::LEFT | Anchor::RIGHT);
        layer.set_size(0, cfg.height); // width 0 + left|right anchors => stretch to the output
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        if cfg.exclusive {
            layer.set_exclusive_zone(cfg.height as i32);
        }
        layer.commit();

        // Pick the real screencopy source if the compositor offers the manager and the
        // user hasn't forced the fallback; otherwise paint the synthetic test pattern.
        let anchor_top = matches!(cfg.anchor, Edge::Top);
        let backdrop = match globals.bind::<ZwlrScreencopyManagerV1, _, _>(qh, 1..=3, ()) {
            Ok(mgr) if !cfg.no_capture => {
                eprintln!("wayshade-panel: capturing backdrop via zwlr_screencopy_manager_v1");
                Backdrop::Screencopy(Box::new(Capture::new(mgr, &shm, cfg.cursor, anchor_top)))
            }
            _ => {
                eprintln!("wayshade-panel: no screencopy manager using synthetic backdrop");
                Backdrop::Synthetic(Synthetic::new())
            }
        };

        // The captured strip is routed through libfx for a live dual-Kawase blur,
        // unless --no-blur asks for the raw mirror. CPU by default; --gpu picks CUDA.
        let blur = if cfg.no_blur {
            eprintln!("wayshade-panel: blur disabled, mirroring the raw strip");
            None
        } else {
            eprintln!(
                "wayshade-panel: {} blur, kawase offset {}",
                if cfg.gpu { "GPU" } else { "CPU" },
                cfg.blur
            );
            Some(Blur::new(cfg.gpu, cfg.blur).expect("create fx blur context"))
        };

        // Enough for a 1080p-wide bar up front; create_buffer grows it for wider outputs.
        let pool = SlotPool::new(1920 * cfg.height as usize * 4, &shm).expect("create shm pool");

        App {
            registry_state: RegistryState::new(globals),
            output_state: OutputState::new(globals, qh),
            shm,
            pool,
            layer,
            backdrop,
            blur,
            width: 0,
            height: cfg.height,
            time: 0,
            cfg,
            configured: false,
            exit: false,
            prev_raw: Vec::new(),
            cached: Vec::new(),
            frames: 0,
            last_report: 0,
            blur_sum: Duration::ZERO,
            blur_max: Duration::ZERO,
            blur_count: 0,
            blur_skipped: 0,
        }
    }

    // Start the next backdrop capture (or repaint the synthetic pattern). The real
    // source returns immediately and lands its pixels async by the next frame, so
    // capture stays pipelined one frame behind present and never blocks here.
    fn kick_capture(&mut self, qh: &QueueHandle<App>) {
        let out = self.output_state.outputs().next();
        let (w, h, t) = (self.width, self.height, self.time);
        match &mut self.backdrop {
            Backdrop::Synthetic(s) => s.fill(t, w, h),
            Backdrop::Screencopy(c) => {
                if let Some(o) = out {
                    c.begin(&o, qh);
                }
            }
        }
    }

    // Format and reset the accumulated blur-time stats for the periodic report.
    // Empty when nothing was drawn (e.g. --no-blur).
    fn drain_blur_stats(&mut self) -> String {
        if self.blur_count == 0 && self.blur_skipped == 0 {
            return String::new();
        }
        let avg = if self.blur_count > 0 {
            self.blur_sum.as_secs_f32() * 1000.0 / self.blur_count as f32
        } else {
            0.0
        };
        let max = self.blur_max.as_secs_f32() * 1000.0;
        let s = format!(
            ", blur {avg:.2}ms avg / {max:.2}ms max ({} blurred, {} skipped)",
            self.blur_count, self.blur_skipped
        );
        self.blur_sum = Duration::ZERO;
        self.blur_max = Duration::ZERO;
        self.blur_count = 0;
        self.blur_skipped = 0;
        s
    }

    fn draw(&mut self, qh: &QueueHandle<App>) {
        let (w, h) = (self.width, self.height);
        if w == 0 || h == 0 {
            return;
        }
        let stride = w as i32 * 4;
        let (buffer, canvas) = self
            .pool
            .create_buffer(w as i32, h as i32, stride, wl_shm::Format::Argb8888)
            .expect("create_buffer");

        // present() leaves the raw mirrored strip in `canvas`; blur it in place and
        // composite the tint over it. With no capture yet, fall back to a flat tint.
        if self.backdrop.present(canvas, w, h) {
            // With --no-blur, blur is None and the raw mirrored strip stays as-is.
            if let Some(blur) = &self.blur {
                if self.prev_raw.as_slice() == &*canvas {
                    // backdrop unchanged: reuse the cached blur, skip the pass
                    canvas.copy_from_slice(&self.cached);
                    self.blur_skipped += 1;
                } else {
                    self.prev_raw.clear();
                    self.prev_raw.extend_from_slice(canvas);
                    let t0 = Instant::now();
                    blur.run(canvas, w, h).expect("blur");
                    composite_tint(canvas, self.cfg.color, self.cfg.alpha);
                    let dt = t0.elapsed();
                    self.blur_sum += dt;
                    self.blur_max = self.blur_max.max(dt);
                    self.blur_count += 1;
                    self.cached.clear();
                    self.cached.extend_from_slice(canvas);
                }
            }
        } else {
            let a = self.cfg.alpha;
            let [r, g, b] = self.cfg.color;
            // wl_shm ARGB8888 is premultiplied, little-endian bytes B,G,R,A.
            let pm = |c: u8| ((c as u16 * a as u16) / 255) as u8;
            let (pb, pg, pr) = (pm(b), pm(g), pm(r));
            for px in canvas.chunks_exact_mut(4) {
                px[0] = pb;
                px[1] = pg;
                px[2] = pr;
                px[3] = a;
            }
        }

        let surface = self.layer.wl_surface();
        surface.frame(qh, surface.clone()); // request the next vsync tick
        surface.damage_buffer(0, 0, w as i32, h as i32);
        buffer.attach_to(surface).expect("attach buffer");
        self.layer.commit();
    }
}

impl CompositorHandler for App {
    fn scale_factor_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: i32) {}
    fn transform_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: wl_output::Transform) {}

    fn frame(&mut self, _: &Connection, qh: &QueueHandle<Self>, _: &wl_surface::WlSurface, time: u32) {
        self.time = time;
        if self.last_report == 0 {
            self.last_report = time;
        }
        self.frames += 1;
        if time.saturating_sub(self.last_report) >= 1000 {
            let cap = match &mut self.backdrop {
                Backdrop::Screencopy(c) => c.drain_stats(),
                Backdrop::Synthetic(_) => None,
            };
            let cap_str = match cap {
                Some((avg, max, n)) => format!(", capture {avg:.2}ms avg / {max:.2}ms max ({n} frames)"),
                None => String::new(),
            };
            let blur_str = self.drain_blur_stats();
            eprintln!("wayshade-panel: {} fps{cap_str}{blur_str}", self.frames);
            self.frames = 0;
            self.last_report = time;
        }
        self.kick_capture(qh);
        self.draw(qh);
    }

    fn surface_enter(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: &wl_output::WlOutput) {}
    fn surface_leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: &wl_output::WlOutput) {}
}

impl LayerShellHandler for App {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {
        self.exit = true;
    }

    fn configure(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        _: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _: u32,
    ) {
        let (w, h) = configure.new_size;
        self.width = if w == 0 { 1920 } else { w };
        self.height = if h == 0 { self.cfg.height } else { h };
        if !self.configured {
            self.configured = true;
            self.draw(qh); // kick off the frame-callback loop
        }
    }
}

impl OutputHandler for App {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, output: wl_output::WlOutput) {
        if let Some(info) = self.output_state.info(&output) {
            eprintln!("wayshade-panel: output added: {}", info.name.unwrap_or_default());
        }
    }

    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl ShmHandler for App {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

// wlr-screencopy isn't covered by SCTK's delegates, so dispatch its two objects by
// hand. The manager is event-less; the frame drives the whole capture lifecycle.
impl Dispatch<ZwlrScreencopyManagerV1, ()> for App {
    fn event(_: &mut Self, _: &ZwlrScreencopyManagerV1, _: <ZwlrScreencopyManagerV1 as wayland_client::Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<ZwlrScreencopyFrameV1, ()> for App {
    fn event(state: &mut Self, _: &ZwlrScreencopyFrameV1, event: zwlr_screencopy_frame_v1::Event, _: &(), _: &Connection, qh: &QueueHandle<Self>) {
        use zwlr_screencopy_frame_v1::{Event, Flags};
        let Backdrop::Screencopy(cap) = &mut state.backdrop else {
            return;
        };
        match event {
            Event::Buffer { format, width, height, stride } => cap.on_buffer(format, width, height, stride, qh),
            Event::Flags { flags } => cap.set_y_invert(matches!(flags, WEnum::Value(f) if f.contains(Flags::YInvert))),
            Event::BufferDone => cap.on_buffer_done(qh),
            Event::Ready { .. } => cap.on_ready(),
            Event::Failed => cap.on_failed(),
            _ => {} // damage / linux_dmabuf: not used for the shm path
        }
    }
}

// Our reused capture buffer. We gate the next copy on the frame's `ready` event, so
// release tracking isn't needed here, this just satisfies the Dispatch bound.
impl Dispatch<wl_buffer::WlBuffer, ()> for App {
    fn event(_: &mut Self, _: &wl_buffer::WlBuffer, _: wl_buffer::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

impl ProvidesRegistryState for App {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

delegate_compositor!(App);
delegate_output!(App);
delegate_shm!(App);
delegate_layer!(App);
delegate_registry!(App);
