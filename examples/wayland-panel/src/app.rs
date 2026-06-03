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
    Connection, QueueHandle,
    globals::GlobalList,
    protocol::{wl_output, wl_shm, wl_surface},
};

use crate::{Config, Edge};

pub struct App {
    registry_state: RegistryState,
    output_state: OutputState,
    shm: Shm,
    pool: SlotPool,
    layer: LayerSurface,
    cfg: Config,
    width: u32,
    height: u32,
    configured: bool,
    pub exit: bool,
    // rough fps accounting, reported once a second to stderr
    frames: u32,
    last_report: u32,
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

        // Enough for a 1080p-wide bar up front; create_buffer grows it for wider outputs.
        let pool = SlotPool::new(1920 * cfg.height as usize * 4, &shm).expect("create shm pool");

        App {
            registry_state: RegistryState::new(globals),
            output_state: OutputState::new(globals, qh),
            shm,
            pool,
            layer,
            width: 0,
            height: cfg.height,
            cfg,
            configured: false,
            exit: false,
            frames: 0,
            last_report: 0,
        }
    }

    fn draw(&mut self, qh: &QueueHandle<App>, time: u32) {
        let (w, h) = (self.width, self.height);
        if w == 0 || h == 0 {
            return;
        }
        let stride = w as i32 * 4;
        let (buffer, canvas) = self
            .pool
            .create_buffer(w as i32, h as i32, stride, wl_shm::Format::Argb8888)
            .expect("create_buffer");

        // Gentle brightness pulse so the vsync-driven frame loop is visibly alive.
        let pulse = (time as f32 / 1000.0 * std::f32::consts::PI).sin() * 0.25 + 0.75;
        let a = self.cfg.alpha as u16;
        let [r, g, b] = self.cfg.color;
        // wl_shm ARGB8888 is premultiplied, little-endian bytes B,G,R,A.
        let pm = |c: u8| ((c as f32 * pulse) as u16 * a / 255) as u8;
        let (pb, pg, pr, pa) = (pm(b), pm(g), pm(r), self.cfg.alpha);
        for px in canvas.chunks_exact_mut(4) {
            px[0] = pb;
            px[1] = pg;
            px[2] = pr;
            px[3] = pa;
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
        if self.last_report == 0 {
            self.last_report = time;
        }
        self.frames += 1;
        if time.saturating_sub(self.last_report) >= 1000 {
            eprintln!("wayshade-panel: {} fps", self.frames);
            self.frames = 0;
            self.last_report = time;
        }
        self.draw(qh, time);
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
            self.draw(qh, 0); // kick off the frame-callback loop
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
