// WayShade demo: a wlr-layer-shell bar. Stage 10 is pure protocol scaffolding —
// a solid (gently pulsing) color bar anchored to an edge, driven by a vsync frame
// loop. No effects yet; the backdrop capture + blur land in later stages.

mod app;

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
    pub exclusive: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            anchor: Edge::Top,
            height: 40,
            color: [0x28, 0x2a, 0x36], // a calm dark slate
            alpha: 224,
            exclusive: false,
        }
    }
}

const USAGE: &str = "\
wayshade-panel — a wlr-layer-shell demo bar

usage: wayshade-panel [--anchor top|bottom] [--height N] [--color RRGGBB]
                      [--alpha 0-255] [--exclusive]

  --anchor    edge to dock the bar to (default top)
  --height    bar height in px (default 40)
  --color     fill color, hex RRGGBB (default 282a36)
  --alpha     bar opacity 0-255 (default 224)
  --exclusive reserve the bar's space so windows don't overlap it
  --help      print this and exit";

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
                "--color" => cfg.color = parse_hex(&val()?)?,
                "--alpha" => cfg.alpha = val()?.parse().map_err(|_| "--alpha must be 0-255".to_string())?,
                "--exclusive" => cfg.exclusive = true,
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
