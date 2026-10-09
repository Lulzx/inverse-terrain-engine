//! Flood simulation videos rendered with fframes on the GPU.
//!
//! `Flood3d`: the no-change terrain (top) and the optimized design (bottom), ray-marched by
//! `shaders/terrain.sksl` from the recorded solver state. One video frame per recorded
//! frame; the overlays are SVG.
pub mod data;

use data::RunData;
use fframes::{AudioMap, Color, Duration, FFramesContext, Frame, Shader, ShaderUniforms, Svgr, Video, include_media_dir};

include_media_dir!(pub struct VideoMedia, "media");

pub const WIDTH: usize = 1920;
pub const HEIGHT: usize = 1080;
const STRIP: f32 = HEIGHT as f32 / 2.0;
const FONT: &str = "DM Sans";
const WEIGHT: u16 = 500;
const FLOOD: &str = "#f0533f";
const SAFE: &str = "#5fd08a";
const INK: &str = "#ffffff";
const MUTED: &str = "#c4cfdb";
/// Seconds held on the final state at the end.
const HOLD_S: f32 = 3.5;

#[derive(Clone, Copy)]
struct V3(f32, f32, f32);
impl V3 {
    fn sub(self, o: V3) -> V3 { V3(self.0 - o.0, self.1 - o.1, self.2 - o.2) }
    fn dot(self, o: V3) -> f32 { self.0 * o.0 + self.1 * o.1 + self.2 * o.2 }
    fn cross(self, o: V3) -> V3 { V3(self.1 * o.2 - self.2 * o.1, self.2 * o.0 - self.0 * o.2, self.0 * o.1 - self.1 * o.0) }
    fn norm(self) -> V3 { let l = self.dot(self).sqrt(); V3(self.0 / l, self.1 / l, self.2 / l) }
}

struct Camera { pos: V3, fwd: V3, right: V3, up: V3, focal: f32 }

impl Camera {
    fn look(pos: V3, target: V3, fov_deg: f32) -> Self {
        let fwd = target.sub(pos).norm();
        let right = fwd.cross(V3(0.0, 0.0, 1.0)).norm();
        let up = right.cross(fwd);
        Self { pos, fwd, right, up, focal: 0.5 / (fov_deg.to_radians() / 2.0).tan() }
    }
    /// World point -> pixel in a viewport of `w` x `h`, or None behind the camera.
    fn project(&self, p: V3, w: f32, h: f32) -> Option<(f32, f32)> {
        let v = p.sub(self.pos);
        let z = v.dot(self.fwd);
        (z > 1.0).then(|| (v.dot(self.right) / z * self.focal * h + 0.5 * w, -v.dot(self.up) / z * self.focal * h + 0.5 * h))
    }
}

/// Shader inputs for view `k` (0 no change, 1 design) at recorded frame `i`.
fn uniforms(data: &RunData, k: usize, i: usize, cam: &Camera, weather: (f32, f32, f32), ve: f32, ortho: bool) -> ShaderUniforms {
    let m = &data.meta;
    let v = &data.views[k];
    let (rain, wet, overcast) = weather;
    ShaderUniforms::new()
        .image("uTerrain", &v.terrain)
        .image("uDepth", &v.depth[i])
        .image("uVel", &v.vel[i])
        .float3("uCamPos", cam.pos.0, cam.pos.1, cam.pos.2)
        .float3("uCamFwd", cam.fwd.0, cam.fwd.1, cam.fwd.2)
        .float3("uCamRight", cam.right.0, cam.right.1, cam.right.2)
        .float3("uCamUp", cam.up.0, cam.up.1, cam.up.2)
        .float("uFocal", cam.focal)
        .float3("uDomain", m.grid.width, m.grid.height, m.grid.cell)
        .float2("uZ", m.z.lo, m.z.span)
        .float("uDepthSpan", m.depth_span)
        .float("uUp", m.grid.up as f32)
        .float("uVE", ve)
        .float("uTime", m.times[i])
        .float("uRain", rain)
        .float("uWet", wet)
        .float("uOvercast", overcast)
        .float("uVelMax", m.vel_max)
        .float("uBase", 9.0)
        .float("uOrtho", if ortho { 1.0 } else { 0.0 })
        .float("uDebug", 0.0)
}

/// Rain rate (0..1 of peak), wetness and cloud cover at recorded frame i; the sky clears
/// over ~5 min after the rain stops and the ground dries over ~30 min.
fn weather(data: &RunData, i: usize) -> (f32, f32, f32) {
    let m = &data.meta;
    let rain = m.rain[i] / m.rain_max.max(1e-6);
    let (mut cum, mut last_rain, mut recent) = (0.0f32, None, 0.0f32);
    for k in 0..=i {
        let dt = if k > 0 { m.times[k] - m.times[k - 1] } else { m.times[0] };
        cum += m.rain[k] / 3600.0 * dt;
        if m.rain[k] > 0.0 {
            last_rain = Some(m.times[k]);
        }
        if m.times[i] - m.times[k] < 300.0 {
            recent = recent.max(m.rain[k] / m.rain_max.max(1e-6));
        }
    }
    let dry = last_rain.map_or(1.0, |t| (-(m.times[i] - t) / 1800.0).exp());
    let wet = (cum / 4.0).clamp(0.0, 1.0) * dry;
    let overcast = (recent * 2.5).clamp(0.0, 1.0) * 0.85 + 0.1;
    (rain, wet, overcast)
}

fn title_case(s: &str) -> String {
    s.split('-').map(|w| {
        let mut c = w.chars();
        c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
    }).collect::<Vec<_>>().join(" ")
}

/// Building that ends below its limit in the design run.
fn saved_count(data: &RunData) -> usize {
    data.meta.houses.iter().enumerate()
        .filter(|(h, house)| data.views[1].series[*h].iter().fold(0.0f32, |a, b| a.max(*b)) <= house.threshold)
        .count()
}

fn result_line(data: &RunData) -> String {
    let m = &data.meta;
    let r = &m.result;
    format!(
        "{} of {} buildings stay below their limit · objective {:.0}% lower · downstream area {:+.1} cm (allowed {:.0} cm) · {:.0} m³ of earth moved",
        saved_count(data), m.houses.len(), (1.0 - r.j_after / r.j_before) * 100.0, r.guard_worsening_m * 100.0,
        r.guard_tolerance_m * 100.0, r.fill_m3 + r.cut_m3,
    )
}

pub struct Flood3d<'a> {
    pub data: &'a RunData,
    shader: Shader,
    title: String,
    vert_exag: f32,
    pub debug: f32,
}

impl<'a> Flood3d<'a> {
    pub fn new(data: &'a RunData, vert_exag: f32) -> Self {
        let title = title_case(&data.meta.example);
        Self { data, shader: Shader::sksl(include_str!("shaders/terrain.sksl")), title, vert_exag, debug: 0.0 }
    }

    fn state(&self, frame: &Frame) -> (usize, f32) {
        let n = self.data.frames();
        let i = frame.global_index.min(n - 1);
        // Hold: 0..1 over the extra frames at the end.
        let hold = (frame.global_index as f32 - (n - 1) as f32) / (HOLD_S * Self::FPS as f32);
        (i, hold.clamp(0.0, 1.0))
    }

    fn camera(&self, frame: &Frame) -> Camera {
        let m = &self.data.meta;
        let (w, h) = (m.grid.width, m.grid.height);
        let total = (self.data.frames() as f32 + HOLD_S * Self::FPS as f32).max(1.0);
        let u = frame.global_index as f32 / total;
        let az = (-22.0 + 30.0 * u).to_radians();
        let el = 27f32.to_radians();
        let dist = 0.66 * (w * w + h * h).sqrt();
        let target = V3(w * 0.5, h * 0.5, self.data.ground_mean * self.vert_exag);
        let pos = V3(target.0 + dist * el.cos() * az.sin(), target.1 - dist * el.cos() * az.cos(), target.2 + dist * el.sin());
        Camera::look(pos, target, 34.0)
    }

    fn view_layer(&self, frame: &Frame, k: usize, i: usize, cam: &Camera, weather: (f32, f32, f32)) -> Svgr<'_> {
        let layer = self.shader.draw(frame, uniforms(self.data, k, i, cam, weather, self.vert_exag, false).float("uDebug", self.debug));
        let y = k as f32 * STRIP;
        fframes::svgr!(<image href={layer.href()} x="0" y={y} width={WIDTH} height={STRIP} />)
    }

    fn strip_overlay(&self, k: usize, i: usize, cam: &Camera) -> Svgr<'_> {
        let m = &self.data.meta;
        let v = &self.data.views[k];
        let y0 = k as f32 * STRIP;
        let label = if k == 0 { "NO CHANGE" } else { "WITH OPTIMIZED EARTHWORKS" };
        let label_w = if k == 0 { 168.0 } else { 392.0 };
        let label_y = if k == 0 { y0 + 112.0 } else { y0 + 22.0 };
        let label_fill = if k == 0 { INK } else { "#ffc88c" };

        // Building pins and depth card.
        let mut pins = Vec::new();
        for (h, (house, top)) in m.houses.iter().zip(&v.house_tops).enumerate() {
            let d = v.series[h][i];
            let over = d > house.threshold;
            let col = if over { FLOOD } else { SAFE };
            let p = V3(top[0], top[1], (top[2] + 1.0) * self.vert_exag);
            if let Some((px, py)) = cam.project(p, WIDTH as f32, STRIP) {
                let py = py + y0;
                let name = house.name.replace('_', " ");
                pins.push(fframes::svgr!(
                    <g>
                        <line x1={px} y1={py} x2={px} y2={py - 26.0} stroke="#ffffff" stroke-opacity="0.8" stroke-width="1.5" />
                        <circle cx={px} cy={py - 30.0} r="6" fill={col} stroke="#0b1422" stroke-width="1.5" />
                        <text x={px + 10.0} y={py - 25.0} font-family={FONT} font-size="15" font-weight={WEIGHT} fill={INK} stroke="#0b1422" stroke-width="3" paint-order="stroke">
                            {name}
                        </text>
                    </g>
                ));
            }
        }

        let rows = m.houses.len() as f32;
        let (cw, ch) = (390.0, 46.0 + 32.0 * rows);
        let (cx, cy) = (WIDTH as f32 - cw - 28.0, y0 + STRIP - ch - 24.0);
        let scale = m.houses.iter().map(|h| h.threshold * 3.0).fold(0.0f32, f32::max)
            .max(self.data.views.iter().flat_map(|v| v.series.iter().flatten()).fold(0.0f32, |a, b| a.max(*b)) * 1.1);
        let mut card_rows = Vec::new();
        for (h, house) in m.houses.iter().enumerate() {
            let d = v.series[h][i];
            let peak = v.series[h][..=i].iter().fold(0.0f32, |a, b| a.max(*b));
            let over = d > house.threshold;
            let col = if over { FLOOD } else { SAFE };
            let ry = cy + 50.0 + 32.0 * h as f32;
            let (bx, bw) = (cx + 130.0, 150.0);
            let fill = (bw * (d / scale).min(1.0)).max(0.5);
            let lim = bx + bw * house.threshold / scale;
            let pk = bx + bw * (peak / scale).min(1.0);
            card_rows.push(fframes::svgr!(
                <g>
                    <circle cx={cx + 22.0} cy={ry} r="6" fill={col} />
                    <text x={cx + 36.0} y={ry + 6.0} font-family={FONT} font-size="17" font-weight={WEIGHT} fill={INK}>{house.name.replace('_', " ")}</text>
                    <rect x={bx} y={ry - 6.0} width={bw} height="12" rx="6" fill="#3a4452" />
                    <rect x={bx} y={ry - 6.0} width={fill} height="12" rx="6" fill={col} />
                    <line x1={pk} y1={ry - 8.0} x2={pk} y2={ry + 8.0} stroke="#9aa6b4" stroke-width="1.5" />
                    <line x1={lim} y1={ry - 10.0} x2={lim} y2={ry + 10.0} stroke="#ffffff" stroke-width="2.5" />
                    <text x={cx + cw - 18.0} y={ry + 6.0} font-family={FONT} font-size="17" font-weight={WEIGHT} fill={col} text-anchor="end">{format!("{:.1} cm", d * 100.0)}</text>
                </g>
            ));
        }

        fframes::svgr!(
            <g>
                <rect x="28" y={label_y} width={label_w} height="40" rx="20" fill="#08101b" fill-opacity="0.72" />
                <text x="48" y={label_y + 27.0} font-family={FONT} font-size="20" font-weight={WEIGHT} fill={label_fill} letter-spacing="1.5">{label}</text>
                {pins}
                <rect x={cx} y={cy} width={cw} height={ch} rx="14" fill="#08101b" fill-opacity="0.72" />
                <text x={cx + 18.0} y={cy + 28.0} font-family={FONT} font-size="14" font-weight={WEIGHT} fill={MUTED} letter-spacing="1">"WATER AT BUILDINGS (white tick: limit)"</text>
                {card_rows}
            </g>
        )
    }

    fn rain_streaks(&self, frame: &Frame, rain: f32) -> Svgr<'_> {
        let n = (rain * 420.0) as usize;
        let mut lines = Vec::with_capacity(n);
        for k in 0..n {
            // Deterministic per-drop position, falling ~55 px per frame.
            let h1 = ((k as f32 * 12.9898).sin() * 43758.545).fract().abs();
            let h2 = ((k as f32 * 78.233).sin() * 12543.17).fract().abs();
            let x = h1 * (WIDTH as f32 + 200.0) - 100.0;
            let y = ((h2 * HEIGHT as f32 + frame.global_index as f32 * 55.0) % (HEIGHT as f32 + 60.0)) - 30.0;
            lines.push(fframes::svgr!(<line x1={x} y1={y} x2={x - 6.0} y2={y + 30.0} stroke="#dbe8ff" stroke-opacity="0.32" stroke-width="1.2" />));
        }
        fframes::svgr!(<g>{lines}</g>)
    }
}

impl Video for Flood3d<'_> {
    const FPS: usize = 30;
    const WIDTH: usize = WIDTH;
    const HEIGHT: usize = HEIGHT;
    const BACKGROUND_COLOR: Color = Color::BLACK;

    fn duration(&self) -> Duration<'_> {
        Duration::Frames(self.data.frames() + (HOLD_S * Self::FPS as f32) as usize)
    }

    fn audio(&self) -> AudioMap<'_> {
        AudioMap::none()
    }

    fn render_frame<'a>(&'a self, frame: Frame, _ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let m = &self.data.meta;
        let (i, hold) = self.state(&frame);
        let cam = self.camera(&frame);
        let weather = weather(self.data, i);
        let top = self.view_layer(&frame, 0, i, &cam, weather);
        let bottom = self.view_layer(&frame, 1, i, &cam, weather);
        let over_top = self.strip_overlay(0, i, &cam);
        let over_bottom = self.strip_overlay(1, i, &cam);
        let rain = self.rain_streaks(&frame, weather.0);

        let t = m.times[i];
        let clock = format!("{:02}:{:02}", (t / 60.0) as u32, (t % 60.0) as u32);
        let rain_mm = m.rain[i];
        let gauge = (286.0 * rain_mm / m.rain_max.max(1e-6)).max(0.5);
        let subtitle = format!(
            "Solver output every {:.0} s, no interpolation · true vertical scale{} · {} × {} cells at {} m · Manning n = {}",
            m.sync_s, if (self.vert_exag - 1.0).abs() > 1e-3 { format!(" ×{}", self.vert_exag) } else { String::new() },
            m.grid.nx, m.grid.ny, m.grid.cell, m.manning_n,
        );

        let banner = result_line(self.data);
        let banner_op = hold.min(0.25) * 4.0;

        fframes::svgr!(
            <svg xmlns="http://www.w3.org/2000/svg" viewBox={format!("0 0 {WIDTH} {HEIGHT}")} width={WIDTH} height={HEIGHT}>
                <defs>
                    <linearGradient id="head" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0" stop-color="#050a12" stop-opacity="0.85" />
                        <stop offset="1" stop-color="#050a12" stop-opacity="0" />
                    </linearGradient>
                </defs>
                {top}
                {bottom}
                {rain}
                <rect x="0" y="0" width={WIDTH} height="120" fill="url(#head)" />
                <rect x="0" y={STRIP - 1.5} width={WIDTH} height="3" fill="#0b1422" />
                <text x="44" y="50" font-family={FONT} font-size="32" font-weight={WEIGHT} fill={INK}>{format!("{}: flood simulation with and without earthworks", self.title)}</text>
                <text x="44" y="82" font-family={FONT} font-size="17" font-weight={WEIGHT} fill={MUTED}>{subtitle}</text>
                <text x="1876" y="52" font-family={FONT} font-size="36" font-weight={WEIGHT} fill={INK} text-anchor="end">{clock}</text>
                <rect x="1590" y="72" width="286" height="12" rx="6" fill="#3a4452" />
                <rect x="1590" y="72" width={gauge} height="12" rx="6" fill="#78beff" />
                <text x="1576" y="83" font-family={FONT} font-size="17" font-weight={WEIGHT} fill={MUTED} text-anchor="end">{format!("rain {rain_mm:.0} mm/h", )}</text>
                {over_top}
                {over_bottom}
                <g opacity={banner_op}>
                    <rect x="200" y={STRIP - 30.0} width="1520" height="60" rx="30" fill="#08101b" fill-opacity="0.85" />
                    <text x="960" y={STRIP + 8.0} font-family={FONT} font-size="22" font-weight={WEIGHT} fill={INK} text-anchor="middle">{banner}</text>
                </g>
            </svg>
        )
    }
}

const SERIES: [&str; 6] = ["#6cb4ff", "#ff8a65", "#7ddc8a", "#d6a4ff", "#ffd166", "#5ee0d6"];
const MAP_W: f32 = 720.0;

/// Map view: both terrains from above with charts of the depth at each building.
pub struct Flood2d<'a> {
    pub data: &'a RunData,
    shader: Shader,
    title: String,
}

impl<'a> Flood2d<'a> {
    pub fn new(data: &'a RunData) -> Self {
        Self { data, shader: Shader::sksl(include_str!("shaders/terrain.sksl")), title: title_case(&data.meta.example) }
    }

    fn map_h(&self) -> f32 {
        MAP_W * self.data.meta.grid.height / self.data.meta.grid.width
    }

    fn map(&self, frame: &Frame, k: usize, i: usize, x0: f32, y0: f32) -> Svgr<'_> {
        let m = &self.data.meta;
        let mh = self.map_h();
        let cam = Camera::look(V3(0.0, 0.0, 1.0), V3(0.0, 1.0, 0.0), 40.0);
        let layer = self.shader.draw(frame, uniforms(self.data, k, i, &cam, weather(self.data, i), 1.0, true));
        let to = |p: &[f32; 2]| format!("{:.1},{:.1}", x0 + p[0] / m.grid.width * MAP_W, y0 + (1.0 - p[1] / m.grid.height) * mh);
        let ring = |xy: &Vec<[f32; 2]>| xy.iter().map(to).collect::<Vec<_>>().join(" ");
        let guards: Vec<Svgr> = m.guards.iter().map(|g| fframes::svgr!(
            <polygon points={ring(&g.xy)} fill="#ffb347" fill-opacity="0.12" stroke="#ffb347" stroke-width="2" stroke-dasharray="7 5" />
        )).collect();
        let works: Vec<Svgr> = if k == 1 {
            m.earthworks.iter().map(|w| fframes::svgr!(
                <polygon points={ring(&w.xy)} fill="none" stroke="#ffffff" stroke-opacity="0.85" stroke-width="1.5" stroke-dasharray="4 3" />
            )).collect()
        } else {
            Vec::new()
        };
        let houses: Vec<Svgr> = m.houses.iter().enumerate().map(|(h, house)| {
            let over = self.data.views[k].series[h][i] > house.threshold;
            let (mut lx, mut ly) = (f32::MIN, f32::MAX);
            for p in &house.xy {
                lx = lx.max(x0 + p[0] / m.grid.width * MAP_W);
                ly = ly.min(y0 + (1.0 - p[1] / m.grid.height) * mh);
            }
            fframes::svgr!(
                <g>
                    <polygon points={ring(&house.xy)} fill="none" stroke={if over { FLOOD } else { SAFE }} stroke-width="2.5" />
                    <text x={lx + 5.0} y={ly + 4.0} font-family={FONT} font-size="14" font-weight={WEIGHT} fill={INK} stroke="#0b1422" stroke-width="3" paint-order="stroke">
                        {house.name.replace('_', " ")}
                    </text>
                </g>
            )
        }).collect();
        let label = if k == 0 { "NO CHANGE" } else { "WITH OPTIMIZED EARTHWORKS" };
        let label_w = if k == 0 { 150.0 } else { 340.0 };
        fframes::svgr!(
            <g>
                <image href={layer.href()} x={x0} y={y0} width={MAP_W} height={mh} />
                <rect x={x0} y={y0} width={MAP_W} height={mh} fill="none" stroke="#2a3442" stroke-width="2" />
                {guards}
                {works}
                {houses}
                <rect x={x0 + 14.0} y={y0 + 14.0} width={label_w} height="34" rx="17" fill="#08101b" fill-opacity="0.75" />
                <text x={x0 + 32.0} y={y0 + 37.0} font-family={FONT} font-size="17" font-weight={WEIGHT} fill={if k == 0 { INK } else { "#ffc88c" }} letter-spacing="1.2">{label}</text>
            </g>
        )
    }

    /// Depth at each building over time: dotted no change, solid design, drawn up to now.
    fn chart(&self, i: usize, x0: f32, y0: f32, w: f32, h: f32) -> Svgr<'_> {
        let m = &self.data.meta;
        let t_end = *m.times.last().unwrap_or(&1.0);
        let dmax = self.data.views.iter().flat_map(|v| v.series.iter().flatten()).fold(0.0f32, |a, b| a.max(*b))
            .max(m.houses.iter().map(|h| h.threshold).fold(0.0, f32::max)) * 1.25;
        let px = |t: f32| x0 + t / t_end * w;
        let py = |d: f32| y0 + h - d / dmax * h;
        let line = |s: &[f32], upto: usize| (0..=upto).map(|k| format!("{:.1},{:.1}", px(m.times[k]), py(s[k]))).collect::<Vec<_>>().join(" ");
        let mut paths = Vec::new();
        for (hn, _) in m.houses.iter().enumerate() {
            let c = SERIES[hn % SERIES.len()];
            let last = m.times.len() - 1;
            paths.push(fframes::svgr!(
                <g>
                    <polyline points={line(&self.data.views[0].series[hn], last)} fill="none" stroke={c} stroke-opacity="0.18" stroke-width="1.5" stroke-dasharray="3 4" />
                    <polyline points={line(&self.data.views[1].series[hn], last)} fill="none" stroke={c} stroke-opacity="0.18" stroke-width="2" />
                    <polyline points={line(&self.data.views[0].series[hn], i)} fill="none" stroke={c} stroke-width="2" stroke-dasharray="3 4" />
                    <polyline points={line(&self.data.views[1].series[hn], i)} fill="none" stroke={c} stroke-width="3" />
                </g>
            ));
        }
        // Rain as a filled profile along the top of the chart.
        let rain_pts = std::iter::once(format!("{:.1},{:.1}", x0, y0))
            .chain(m.times.iter().zip(&m.rain).map(|(t, r)| format!("{:.1},{:.1}", px(*t), y0 + r / m.rain_max.max(1e-6) * 46.0)))
            .chain(std::iter::once(format!("{:.1},{:.1}", x0 + w, y0)))
            .collect::<Vec<_>>().join(" ");
        let thr = m.houses.first().map(|h| h.threshold).unwrap_or(0.0);
        let cx = px(m.times[i]);
        let ticks: Vec<Svgr> = (0..=(t_end / 300.0) as usize).map(|k| {
            let t = k as f32 * 300.0;
            fframes::svgr!(<text x={px(t)} y={y0 + h + 24.0} font-family={FONT} font-size="14" font-weight={WEIGHT} fill={MUTED} text-anchor="middle">{format!("{} min", (t / 60.0) as u32)}</text>)
        }).collect();
        let legend: Vec<Svgr> = m.houses.iter().enumerate().map(|(hn, house)| {
            let lx = x0 + 470.0 + 130.0 * hn as f32;
            fframes::svgr!(
                <g>
                    <rect x={lx} y={y0 - 20.0} width="22" height="4" fill={SERIES[hn % SERIES.len()]} />
                    <text x={lx + 30.0} y={y0 - 12.0} font-family={FONT} font-size="15" font-weight={WEIGHT} fill={INK}>{house.name.replace('_', " ")}</text>
                </g>
            )
        }).collect();
        fframes::svgr!(
            <g>
                <rect x={x0} y={y0} width={w} height={h} fill="#0f1823" stroke="#2a3442" stroke-width="1.5" />
                <polygon points={rain_pts} fill="#78beff" fill-opacity="0.22" />
                <line x1={x0} y1={py(thr)} x2={x0 + w} y2={py(thr)} stroke="#ffffff" stroke-opacity="0.75" stroke-width="1.5" stroke-dasharray="8 6" />
                <text x={x0 + w - 10.0} y={py(thr) - 8.0} font-family={FONT} font-size="15" font-weight={WEIGHT} fill={INK} text-anchor="end">{format!("flood limit {:.0} cm", thr * 100.0)}</text>
                {paths}
                <line x1={cx} y1={y0} x2={cx} y2={y0 + h} stroke="#ffffff" stroke-opacity="0.6" stroke-width="1.5" />
                {ticks}
                {legend}
                <text x={x0 + w} y={y0 - 12.0} font-family={FONT} font-size="15" font-weight={WEIGHT} fill={MUTED} text-anchor="end">"dotted: no change · solid: with earthworks · blue: rain"</text>
                <text x={x0} y={y0 - 12.0} font-family={FONT} font-size="17" font-weight={WEIGHT} fill={MUTED} letter-spacing="1">"DEEPEST WATER ON EACH BUILDING FOOTPRINT"</text>
            </g>
        )
    }

    fn cards(&self, i: usize, x0: f32, y0: f32) -> Svgr<'_> {
        let m = &self.data.meta;
        let rows: Vec<Svgr> = m.houses.iter().enumerate().map(|(hn, house)| {
            let ry = y0 + 84.0 + 58.0 * hn as f32;
            let vals: Vec<Svgr> = (0..2).map(|k| {
                let d = self.data.views[k].series[hn][i];
                let peak = self.data.views[k].series[hn][..=i].iter().fold(0.0f32, |a, b| a.max(*b));
                let col = if d > house.threshold { FLOOD } else { SAFE };
                let vx = x0 + 150.0 + 100.0 * k as f32;
                fframes::svgr!(
                    <g>
                        <text x={vx} y={ry} font-family={FONT} font-size="20" font-weight={WEIGHT} fill={col} text-anchor="end">{format!("{:.1}", d * 100.0)}</text>
                        <text x={vx} y={ry + 20.0} font-family={FONT} font-size="13" font-weight={WEIGHT} fill={MUTED} text-anchor="end">{format!("peak {:.1}", peak * 100.0)}</text>
                    </g>
                )
            }).collect();
            fframes::svgr!(
                <g>
                    <circle cx={x0 + 16.0} cy={ry - 6.0} r="6" fill={SERIES[hn % SERIES.len()]} />
                    <text x={x0 + 30.0} y={ry} font-family={FONT} font-size="18" font-weight={WEIGHT} fill={INK}>{house.name.replace('_', " ")}</text>
                    {vals}
                </g>
            )
        }).collect();
        let h = 100.0 + 58.0 * m.houses.len() as f32;
        fframes::svgr!(
            <g>
                <rect x={x0 - 14.0} y={y0} width="300" height={h} rx="14" fill="#0f1823" stroke="#2a3442" stroke-width="1.5" />
                <text x={x0} y={y0 + 26.0} font-family={FONT} font-size="14" font-weight={WEIGHT} fill={MUTED} letter-spacing="1">"WATER DEPTH (cm)"</text>
                <text x={x0 + 150.0} y={y0 + 48.0} font-family={FONT} font-size="14" font-weight={WEIGHT} fill={MUTED} text-anchor="end">"no change"</text>
                <text x={x0 + 250.0} y={y0 + 48.0} font-family={FONT} font-size="14" font-weight={WEIGHT} fill="#ffc88c" text-anchor="end">"design"</text>
                {rows}
                <g transform={format!("translate({x0} {})", y0 + h + 24.0)}>
                    <rect x="0" y="0" width="26" height="16" fill="#ffb347" fill-opacity="0.15" stroke="#ffb347" stroke-width="2" stroke-dasharray="5 4" />
                    <text x="38" y="13" font-family={FONT} font-size="15" font-weight={WEIGHT} fill={INK}>"downstream area"</text>
                    <text x="38" y="32" font-family={FONT} font-size="13" font-weight={WEIGHT} fill={MUTED}>"must not get deeper"</text>
                    <rect x="0" y="50" width="26" height="12" fill="none" stroke="#ffffff" stroke-width="1.5" stroke-dasharray="4 3" />
                    <text x="38" y="62" font-family={FONT} font-size="15" font-weight={WEIGHT} fill={INK}>"earthwork outline"</text>
                    <rect x="0" y="82" width="26" height="16" fill="none" stroke={FLOOD} stroke-width="2.5" />
                    <text x="38" y="95" font-family={FONT} font-size="15" font-weight={WEIGHT} fill={INK}>"building above its limit"</text>
                    <rect x="0" y="112" width="26" height="16" fill="none" stroke={SAFE} stroke-width="2.5" />
                    <text x="38" y="125" font-family={FONT} font-size="15" font-weight={WEIGHT} fill={INK}>"building below its limit"</text>
                    <rect x="0" y="144" width="26" height="16" fill="#5a4630" />
                    <text x="38" y="157" font-family={FONT} font-size="15" font-weight={WEIGHT} fill={INK}>"flood water (silty)"</text>
                </g>
            </g>
        )
    }
}

impl Video for Flood2d<'_> {
    const FPS: usize = 30;
    const WIDTH: usize = WIDTH;
    const HEIGHT: usize = HEIGHT;
    const BACKGROUND_COLOR: Color = Color::BLACK;

    fn duration(&self) -> Duration<'_> {
        Duration::Frames(self.data.frames() + (HOLD_S * Self::FPS as f32) as usize)
    }

    fn audio(&self) -> AudioMap<'_> {
        AudioMap::none()
    }

    fn render_frame<'a>(&'a self, frame: Frame, _ctx: &FFramesContext<'a, '_>) -> Svgr<'a> {
        let m = &self.data.meta;
        let n = self.data.frames();
        let i = frame.global_index.min(n - 1);
        let hold = ((frame.global_index as f32 - (n - 1) as f32) / (HOLD_S * Self::FPS as f32)).clamp(0.0, 1.0);
        let my = 108.0;
        let left = self.map(&frame, 0, i, 40.0, my);
        let right = self.map(&frame, 1, i, 40.0 + MAP_W + 24.0, my);
        let chart_y = my + self.map_h() + 62.0;
        let chart = self.chart(i, 40.0, chart_y, WIDTH as f32 - 80.0, HEIGHT as f32 - chart_y - 44.0);
        let cards = self.cards(i, 40.0 + 2.0 * MAP_W + 62.0, my);
        let t = m.times[i];
        let clock = format!("{:02}:{:02}", (t / 60.0) as u32, (t % 60.0) as u32);
        let gauge = (286.0 * m.rain[i] / m.rain_max.max(1e-6)).max(0.5);
        let banner_op = hold.min(0.25) * 4.0;
        fframes::svgr!(
            <svg xmlns="http://www.w3.org/2000/svg" viewBox={format!("0 0 {WIDTH} {HEIGHT}")} width={WIDTH} height={HEIGHT}>
                <rect width={WIDTH} height={HEIGHT} fill="#0b1422" />
                <text x="40" y="48" font-family={FONT} font-size="32" font-weight={WEIGHT} fill={INK}>{format!("{}: flood maps with and without earthworks", self.title)}</text>
                <text x="40" y="80" font-family={FONT} font-size="17" font-weight={WEIGHT} fill={MUTED}>{format!("Solver output every {:.0} s, no interpolation · {} × {} cells at {} m · orange: downstream area that must not get worse", m.sync_s, m.grid.nx, m.grid.ny, m.grid.cell)}</text>
                <text x="1880" y="50" font-family={FONT} font-size="36" font-weight={WEIGHT} fill={INK} text-anchor="end">{clock}</text>
                <rect x="1594" y="68" width="286" height="12" rx="6" fill="#3a4452" />
                <rect x="1594" y="68" width={gauge} height="12" rx="6" fill="#78beff" />
                <text x="1580" y="79" font-family={FONT} font-size="17" font-weight={WEIGHT} fill={MUTED} text-anchor="end">{format!("rain {:.0} mm/h", m.rain[i])}</text>
                {left}
                {right}
                {cards}
                {chart}
                <g opacity={banner_op}>
                    <rect x="200" y={my + self.map_h() / 2.0 - 30.0} width="1520" height="60" rx="30" fill="#08101b" fill-opacity="0.88" />
                    <text x="960" y={my + self.map_h() / 2.0 + 8.0} font-family={FONT} font-size="22" font-weight={WEIGHT} fill={INK} text-anchor="middle">{result_line(self.data)}</text>
                </g>
            </svg>
        )
    }
}
