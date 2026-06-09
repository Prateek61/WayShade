// WayShade demo: a wlr-layer-shell bar.

mod app;
mod backdrop;
mod effect;

use std::time::Duration;

use calloop::{
    EventLoop,
    signals::{Signal, Signals},
};
use calloop_wayland_source::WaylandSource;
use wayland_client::{Connection, globals::registry_queue_init};

use app::App;

#[derive(Clone, Copy)]
pub enum Edge {
    Top,
    Bottom,
}

#[derive(Clone)]
pub struct Config {
    pub anchor: Edge,
    pub height: u32,
    pub color: [u8; 3],
    pub alpha: u8,
    pub blur: f32,
    pub gpu: bool,
    pub no_blur: bool,
    pub exclusive: bool,
    pub no_capture: bool,
    pub cursor: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            anchor: Edge::Top,
            height: 40,
            color: [0x28, 0x2a, 0x36], // a calm dark slate, used as the frosted tint
            alpha: 140,                // tint strength: low enough to show the blur through it
            blur: 3.0,
            gpu: false,
            no_blur: false,
            exclusive: false,
            no_capture: false,
            cursor: false,
        }
    }
}

const USAGE: &str = "\
wayshade-panel a wlr-layer-shell demo bar with live backdrop blur

usage: wayshade-panel [--anchor top|bottom] [--height N] [--blur OFFSET] [--gpu]
                      [--no-blur] [--color RRGGBB] [--alpha 0-255] [--exclusive]
                      [--no-capture] [--cursor]

  --anchor     edge to dock the bar to (default top)
  --height     bar height in px (default 40)
  --blur       dual-Kawase strength; higher is blurrier (default 3.0)
  --gpu        run the blur on CUDA instead of the CPU
  --no-blur    show the raw captured strip, skipping the blur
  --color      frosted tint over the blurred backdrop, hex RRGGBB (default 282a36)
  --alpha      tint strength 0-255; lower shows more backdrop (default 140)
  --exclusive  reserve the bar's space so windows don't overlap it
  --no-capture force the synthetic backdrop instead of wlr-screencopy
  --cursor     composite the cursor into the captured backdrop
  --help       print this and exit";

impl Config {
    fn from_args() -> Result<Config, String> {
        let mut cfg = Config::default();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut val = || args.next().ok_or_else(|| format!("{arg} needs a value"));
            match arg.as_str() {
                "--anchor" => {
                    cfg.anchor = match val()?.as_str() {
                        "top" => Edge::Top,
                        "bottom" => Edge::Bottom,
                        other => return Err(format!("--anchor expects top|bottom, got '{other}'")),
                    }
                }
                "--height" => cfg.height = val()?.parse().map_err(|_| "--height must be a number".to_string())?,
                "--blur" => cfg.blur = val()?.parse().map_err(|_| "--blur must be a number".to_string())?,
                "--gpu" => cfg.gpu = true,
                "--no-blur" => cfg.no_blur = true,
                "--color" => cfg.color = parse_hex(&val()?)?,
                "--alpha" => cfg.alpha = val()?.parse().map_err(|_| "--alpha must be 0-255".to_string())?,
                "--exclusive" => cfg.exclusive = true,
                "--no-capture" => cfg.no_capture = true,
                "--cursor" => cfg.cursor = true,
                "--help" | "-h" => {
                    println!("{USAGE}");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument '{other}' (try --help)")),
            }
        }
        if cfg.height == 0 {
            return Err("--height must be > 0".into());
        }
        if !cfg.no_blur && cfg.blur <= 0.0 {
            return Err("--blur must be > 0".into());
        }
        Ok(cfg)
    }
}

fn parse_hex(s: &str) -> Result<[u8; 3], String> {
    let s = s.trim_start_matches('#');
    if s.len() != 6 {
        return Err(format!("--color expects RRGGBB, got '{s}'"));
    }
    let byte = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| format!("invalid hex in --color: '{s}'"));
    Ok([byte(0)?, byte(2)?, byte(4)?])
}

fn main() {
    let cfg = Config::from_args().unwrap_or_else(|e| {
        eprintln!("wayshade-panel: {e}");
        std::process::exit(2);
    });

    let conn = Connection::connect_to_env()
        .expect("failed to connect to Wayland (is WAYLAND_DISPLAY set?)");
    let (globals, event_queue) = registry_queue_init::<App>(&conn).expect("registry init failed");
    let qh = event_queue.handle();

    let mut app = App::new(&globals, &qh, cfg);

    // calloop drives the wayland fd plus a SIGINT/SIGTERM source, so Ctrl-C unwinds
    // cleanly (Drop tears down the surface) instead of the default hard kill.
    let mut event_loop = EventLoop::<App>::try_new().expect("event loop");
    let lh = event_loop.handle();
    WaylandSource::new(conn.clone(), event_queue)
        .insert(lh.clone())
        .expect("insert wayland source");

    let signals = Signals::new(&[Signal::SIGINT, Signal::SIGTERM]).expect("signals");
    lh.insert_source(signals, |_, _, app: &mut App| app.exit = true)
        .expect("insert signal source");

    let signal = event_loop.get_signal();
    event_loop
        .run(Duration::from_millis(1000), &mut app, move |app| {
            if app.exit {
                signal.stop();
            }
        })
        .expect("event loop run failed");
}
