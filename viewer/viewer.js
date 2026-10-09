// itr static viewer (spec §11): WebGL2, no framework, no build step.
// Three synchronized maps (before / after / difference) styled as journal-figure panels,
// time scrubber, discharge arrows, asset and earthwork inspection, newly-worsened overlay.
"use strict";

const D = window.ITR_DATA;

// ---------- decoding ----------
function b64(s) {
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
// LZ4 block with u32 little-endian uncompressed-size prefix (lz4_flex compress_prepend_size).
function lz4(src) {
  const n = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24);
  const dst = new Uint8Array(n);
  let i = 4, o = 0;
  while (i < src.length) {
    const tok = src[i++];
    let lit = tok >> 4;
    if (lit === 15) { let b; do { b = src[i++]; lit += b; } while (b === 255); }
    dst.set(src.subarray(i, i + lit), o); i += lit; o += lit;
    if (i >= src.length) break;
    const off = src[i] | (src[i + 1] << 8); i += 2;
    let len = tok & 15;
    if (len === 15) { let b; do { b = src[i++]; len += b; } while (b === 255); }
    len += 4;
    for (let k = 0; k < len; k++) { dst[o] = dst[o - off]; o++; }
  }
  return dst;
}
const f32 = s => new Float32Array(lz4(b64(s)).buffer);
const depthFrame = s => { const u = new Uint16Array(lz4(b64(s)).buffer), f = new Float32Array(u.length); for (let i = 0; i < u.length; i++) f[i] = u[i] / 1000; return f; };

const nx = D.nx, ny = D.ny, geo = D.geo;
const R = {};
for (const [k, v] of Object.entries(D.raster_data || {})) { const a = f32(v); for (let i = 0; i < a.length; i++) if (a[i] === -9999) a[i] = NaN; R[k] = a; }
const opt = D.kind === "optimization";
const runs = opt ? { before: D.before, after: D.after } : { before: D.baseline };
const frames = {};
for (const [k, r] of Object.entries(runs)) if (r) frames[k] = r.frames.map(f => ({ t: f.t, sat: f.saturated, data: f.data }));
const cache = {};
function frame(run, i) {
  const key = run + i;
  if (!cache[key]) cache[key] = depthFrame(frames[run][i].data);
  return cache[key];
}
const vel = {};
for (const [k, r] of Object.entries(runs)) if (r && r.velocity && r.velocity.data) vel[k] = { ...r.velocity, field: f32(r.velocity.data) };


// ---------- colormaps (shared by the shader and the colorbars) ----------
// Depth: ColorBrewer Blues, log scale 5 mm - 2 m. Difference: ColorBrewer RdBu (blue = lower, red = higher), +-0.2 m.
const D_MIN = 0.005, D_MAX = 2, DIFF_MAX = 0.2;
const DEPTH_RAMP = ["#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"];
const DIFF_RAMP = ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"];
const rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
const depthT = d => Math.min(1, Math.max(0, Math.log(d / D_MIN) / Math.log(D_MAX / D_MIN)));
function rampJS(ramp, t) {
  const x = Math.min(1, Math.max(0, t)) * (ramp.length - 1), i = Math.min(Math.floor(x), ramp.length - 2), f = x - i;
  const a = rgb(ramp[i]), b = rgb(ramp[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(255 * (v + (b[k] - v) * f))).join(",")})`;
}
function glslRamp(name, ramp) {
  const n = ramp.length, v = ramp.map(h => `vec3(${rgb(h).map(x => x.toFixed(4)).join(",")})`).join(",");
  return `const vec3 ${name}C[${n}] = vec3[${n}](${v});
vec3 ${name}(float t){ float x = clamp(t, 0., 1.) * ${(n - 1).toFixed(1)}; int i = min(int(floor(x)), ${n - 2}); return mix(${name}C[i], ${name}C[i + 1], x - float(i)); }`;
}

// ---------- WebGL ----------
const VS = `#version 300 es
in vec2 p; uniform vec4 view; uniform vec2 grid; out vec2 cell;
void main(){ cell = p * grid; vec2 s = (cell - view.xy) * view.z; gl_Position = vec4(s.x, -s.y * view.w, 0., 1.); }`;
const FS = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 cell; out vec4 o;
uniform sampler2D z, h, dz; uniform int mode; uniform float vex, cs, worse; uniform vec2 grid;
${glslRamp("depthRamp", DEPTH_RAMP)}
${glslRamp("diffRamp", DIFF_RAMP)}
float Z(ivec2 c){ c = clamp(c, ivec2(0), ivec2(grid) - 1); float v = texelFetch(z, c, 0).r; return isnan(v) ? 1e9 : v; }
vec3 depthCol(float d){ return depthRamp(log(d / ${D_MIN.toFixed(4)}) / log(${(D_MAX / D_MIN).toFixed(1)})); }
vec3 div(float d){ return diffRamp(clamp(d / ${DIFF_MAX.toFixed(2)}, -1., 1.) * .5 + .5); }
void main(){
  if (cell.x < 0. || cell.y < 0. || cell.x >= grid.x || cell.y >= grid.y) { o = vec4(1.); return; }
  ivec2 c = ivec2(cell);
  float zc = texelFetch(z, c, 0).r;
  if (isnan(zc) || zc > 1e8) { o = vec4(.87,.87,.87,1.); return; }
  float gx = (Z(c + ivec2(1,0)) - Z(c - ivec2(1,0))) / (2. * cs) * vex;
  float gy = (Z(c + ivec2(0,1)) - Z(c - ivec2(0,1))) / (2. * cs) * vex;
  if (abs(gx) > 1e6) gx = 0.; if (abs(gy) > 1e6) gy = 0.;
  vec3 n = normalize(vec3(-gx, gy, 1.));
  float shade = clamp(dot(n, normalize(vec3(-.5, .5, .7))), 0., 1.);
  vec3 col = vec3(.3 + .6 * shade);
  float v = texelFetch(h, c, 0).r;
  if (mode == 0) { if (v > ${D_MIN.toFixed(4)}) col = depthCol(v); }
  else if (abs(v) > .005) col = div(v);
  if (worse > .5) { float d = texelFetch(dz, c, 0).r; if (d > .01 && fract((cell.x + cell.y) * .5) < .5) col = mix(col, vec3(0.), .7); }
  o = vec4(col, 1.);
}`;

const $ = id => document.getElementById(id);
function h(tag, props, ...kids) {
  const e = document.createElement(tag);
  if (props) for (const [k, v] of Object.entries(props)) k === "class" ? (e.className = v) : e.setAttribute(k, v);
  for (const c of kids) if (c != null) e.append(c);
  return e;
}
const MINUS = "−";
const fmt = (x, d = 3) => (x == null || isNaN(x)) ? "n/a" : (x < 0 ? MINUS : "") + Math.abs(x).toFixed(d);
const sfmt = (x, d = 3) => (x == null || isNaN(x)) ? "n/a" : (x < 0 ? MINUS : "+") + Math.abs(x).toFixed(d);

const ML = 46, MB = 34; // css px reserved left / below the map frame for axis labels

function mkMap(name, label) {
  const fig = h("div", { class: "fig" });
  const title = h("div", { class: "title" });
  const lm = /^(\([a-z]\))\s*(.*)$/.exec(label);
  title.append(h("b", null, lm[1]), " " + lm[2]);
  const plot = h("div", { class: "plot" });
  const el = h("div", { class: "map" });
  el.style.aspectRatio = `${nx} / ${ny}`;
  const cv = h("canvas"), ov = h("canvas"), ax = h("canvas", { class: "ax" }), cb = h("canvas", { class: "cb" });
  el.append(cv, ov); plot.append(el, ax); fig.append(title, plot, cb); $("maps").append(fig);
  const gl = cv.getContext("webgl2");
  if (!gl) { title.append(" (WebGL2 unavailable)"); return null; }
  const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(x); return x; };
  const pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(pr);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0,0, 1,0, 0,1, 1,1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(pr, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const tex = () => { const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); return t; };
  const T = { z: tex(), h: tex(), dz: tex() };
  const up = (t, a) => { gl.bindTexture(gl.TEXTURE_2D, t); gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, nx, ny, 0, gl.RED, gl.FLOAT, a); };
  up(T.dz, R.delta_peak || new Float32Array(nx * ny));
  return { name, el, cv, ov, ax, cb, gl, pr, T, up, mode: 0, cbCaption: "" };
}

function draw(m) {
  const { gl, pr, T, cv } = m;
  const dpr = devicePixelRatio;
  const w = Math.round(cv.clientWidth * dpr), hh = Math.round(cv.clientHeight * dpr);
  if (cv.width !== w || cv.height !== hh) { cv.width = w; cv.height = hh; m.ov.width = w; m.ov.height = hh; }
  gl.viewport(0, 0, w, hh); gl.useProgram(pr);
  const sc = 2 / (V.span), aspect = w / hh;
  gl.uniform4f(gl.getUniformLocation(pr, "view"), V.cx, V.cy, sc / Math.max(aspect, 1), aspect);
  gl.uniform2f(gl.getUniformLocation(pr, "grid"), nx, ny);
  gl.uniform1i(gl.getUniformLocation(pr, "mode"), m.mode);
  gl.uniform1f(gl.getUniformLocation(pr, "vex"), +$("vex").value);
  gl.uniform1f(gl.getUniformLocation(pr, "cs"), geo.dx);
  gl.uniform1f(gl.getUniformLocation(pr, "worse"), m.name === "after" && $("worse").checked ? 1 : 0);
  [["z", 0], ["h", 1], ["dz", 2]].forEach(([k, i]) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, T[k]); gl.uniform1i(gl.getUniformLocation(pr, k), i); });
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  overlay(m);
  axes(m);
}

// View: centre (cells) and span (cells across the shorter canvas side); the default fits the grid to the frame.
const V = { cx: nx / 2, cy: ny / 2, span: Math.min(nx, ny) };
function toScreen(m, x, y) { // cell coords → overlay px
  const w = m.ov.width, h = m.ov.height, sc = 2 / V.span, aspect = w / h;
  const sx = (x - V.cx) * sc / Math.max(aspect, 1), sy = -(y - V.cy) * sc / Math.max(aspect, 1) * aspect;
  return [(sx + 1) / 2 * w, (1 - sy) / 2 * h];
}
function toCell(m, px, py) {
  const w = m.ov.width, h = m.ov.height, sc = 2 / V.span, aspect = w / h, k = sc / Math.max(aspect, 1);
  const sx = px / w * 2 - 1, sy = 1 - py / h * 2;
  return [sx / k + V.cx, -sy / (k * aspect) + V.cy];
}
const worldToCell = (x, y) => [(x - geo.origin_x) / geo.dx, (geo.origin_y - y) / geo.dy];
const niceStep = t => { const p = Math.pow(10, Math.floor(Math.log10(t))), r = t / p; return (r <= 1 ? 1 : r <= 2 ? 2 : r <= 5 ? 5 : 10) * p; };
const pxPerMetre = m => (toScreen(m, 1, 0)[0] - toScreen(m, 0, 0)[0]) / geo.dx; // device px

// Boundary edges (cell coords) of cells whose peak depth grew by more than 1 cm; built once.
let worseEdges = null;
function worsenedEdges() {
  if (worseEdges) return worseEdges;
  worseEdges = [];
  const d = R.delta_peak; if (!d) return worseEdges;
  const w = (x, y) => x >= 0 && y >= 0 && x < nx && y < ny && d[y * nx + x] > 0.01;
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (!w(x, y)) continue;
    if (!w(x - 1, y)) worseEdges.push(x, y, x, y + 1);
    if (!w(x + 1, y)) worseEdges.push(x + 1, y, x + 1, y + 1);
    if (!w(x, y - 1)) worseEdges.push(x, y, x + 1, y);
    if (!w(x, y + 1)) worseEdges.push(x, y + 1, x + 1, y + 1);
  }
  return worseEdges;
}

function haloText(c, s, x, y, dpr) {
  c.lineJoin = "round"; c.lineWidth = 3 * dpr; c.strokeStyle = "#fff"; c.strokeText(s, x, y);
  c.fillStyle = "#000"; c.fillText(s, x, y);
}

function overlay(m) {
  const dpr = devicePixelRatio;
  const c = m.ov.getContext("2d"); c.clearRect(0, 0, m.ov.width, m.ov.height);
  c.lineWidth = 1.5 * dpr; c.lineCap = "butt";
  // assets: outline = edges between member and non-member cells
  c.strokeStyle = "#000"; c.beginPath();
  for (const a of D.assets || []) {
    if (!a._set) a._set = new Set(a.cells);
    for (const k of a.cells) {
      const x = k % nx, y = Math.floor(k / nx);
      const edge = (dx, dy, x0, y0, x1, y1) => {
        const nxk = x + dx, nyk = y + dy;
        if (nxk >= 0 && nyk >= 0 && nxk < nx && nyk < ny && a._set.has(nyk * nx + nxk)) return;
        const p = toScreen(m, x0, y0), q = toScreen(m, x1, y1); c.moveTo(p[0], p[1]); c.lineTo(q[0], q[1]);
      };
      edge(-1, 0, x, y, x, y + 1); edge(1, 0, x + 1, y, x + 1, y + 1);
      edge(0, -1, x, y, x + 1, y); edge(0, 1, x, y + 1, x + 1, y + 1);
    }
  }
  c.stroke();
  // newly worsened: thin black outline around the hatched region
  if (m.name === "after" && $("worse").checked) {
    const e = worsenedEdges();
    c.lineWidth = 1 * dpr; c.strokeStyle = "#000"; c.beginPath();
    for (let i = 0; i < e.length; i += 4) { const p = toScreen(m, e[i], e[i + 1]), q = toScreen(m, e[i + 2], e[i + 3]); c.moveTo(p[0], p[1]); c.lineTo(q[0], q[1]); }
    c.stroke();
  }
  // earthworks
  if (D.earthworks && m.name !== "before") {
    c.strokeStyle = "#8b0000"; c.lineWidth = 1.5 * dpr; c.setLineDash([5 * dpr, 3 * dpr]);
    for (const f of D.earthworks.features) {
      c.beginPath();
      f.geometry.coordinates[0].forEach(([x, y], i) => { const [cx, cy] = worldToCell(x, y); const [sx, sy] = toScreen(m, cx, cy); i ? c.lineTo(sx, sy) : c.moveTo(sx, sy); });
      c.stroke();
    }
    c.setLineDash([]);
  }
  // arrows
  const run = m.name === "diff" ? null : m.name;
  if (run && vel[run] && $("arrows").checked && S.layer === "depth") {
    const v = vel[run], per = v.nx * v.ny * 2, off = S.t * per, st = v.stride;
    c.strokeStyle = "rgba(30,30,30,.85)"; c.lineWidth = 1 * dpr;
    const px = toScreen(m, 1, 0)[0] - toScreen(m, 0, 0)[0];
    // Normalize by the 95th percentile of wet-cell |q| so a few outlet cells don't shrink every arrow.
    const mags = [];
    for (let k = 0; k < per; k += 2) { const q = Math.hypot(v.field[off + k], v.field[off + k + 1]); if (q > 1e-6) mags.push(q); }
    mags.sort((a, b) => a - b);
    const qmax = mags.length ? mags[Math.floor(mags.length * 0.95)] : 1;
    for (let j = 0; j < v.ny; j++) for (let i = 0; i < v.nx; i++) {
      const qx = v.field[off + 2 * (j * v.nx + i)], qy = v.field[off + 2 * (j * v.nx + i) + 1];
      const q = Math.hypot(qx, qy); if (q < 0.02 * qmax) continue;
      const L = Math.min(1, Math.sqrt(q / qmax)) * st * px * 0.9;  // length ∝ √(q/q_max) per frame
      const [x0, y0] = toScreen(m, st / 2 + i * st + 0.5, st / 2 + j * st + 0.5);
      const ux = qx / q, uy = -qy / q, x1 = x0 + ux * L, y1 = y0 + uy * L;
      c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1);
      c.lineTo(x1 - (ux * .35 - uy * .2) * L, y1 - (uy * .35 + ux * .2) * L); c.stroke();
    }
  }
  // scale bar (bottom left) and north arrow (top right)
  const ppm = pxPerMetre(m) / dpr; // css px per metre
  if (ppm > 0) {
    const len = niceStep(90 / ppm), bw = len * ppm * dpr, x0 = 12 * dpr, y0 = m.ov.height - 14 * dpr;
    c.lineWidth = 2 * dpr; c.strokeStyle = "#fff"; c.beginPath(); c.moveTo(x0, y0); c.lineTo(x0 + bw, y0); c.stroke();
    c.lineWidth = 1.5 * dpr; c.strokeStyle = "#000"; c.beginPath();
    c.moveTo(x0, y0 - 4 * dpr); c.lineTo(x0, y0); c.lineTo(x0 + bw, y0); c.lineTo(x0 + bw, y0 - 4 * dpr); c.stroke();
    c.font = `${11 * dpr}px Helvetica, Arial, sans-serif`; c.textAlign = "left"; c.textBaseline = "bottom";
    haloText(c, `${len} m`, x0, y0 - 5 * dpr, dpr);
  }
  const nxp = m.ov.width - 16 * dpr, ny0 = 32 * dpr, ny1 = 12 * dpr + 16 * dpr;
  c.lineWidth = 3.5 * dpr; c.strokeStyle = "#fff"; c.beginPath(); c.moveTo(nxp, ny0 + 8 * dpr); c.lineTo(nxp, ny1 - 2 * dpr); c.stroke();
  c.lineWidth = 1.5 * dpr; c.strokeStyle = "#000"; c.fillStyle = "#000";
  c.beginPath(); c.moveTo(nxp, ny0 + 8 * dpr); c.lineTo(nxp, ny1); c.stroke();
  c.beginPath(); c.moveTo(nxp, ny1 - 3 * dpr); c.lineTo(nxp - 4 * dpr, ny1 + 6 * dpr); c.lineTo(nxp + 4 * dpr, ny1 + 6 * dpr); c.closePath(); c.fill();
  c.font = `bold ${11 * dpr}px Helvetica, Arial, sans-serif`; c.textAlign = "center"; c.textBaseline = "bottom";
  haloText(c, "N", nxp, ny1 - 3 * dpr, dpr);
}

// Axis ticks: easting / northing in metres from the south-west corner of the grid, drawn outside the frame.
function axes(m) {
  const dpr = devicePixelRatio, a = m.ax;
  const aw = Math.round(a.clientWidth * dpr), ah = Math.round(a.clientHeight * dpr);
  if (a.width !== aw || a.height !== ah) { a.width = aw; a.height = ah; }
  const c = a.getContext("2d"); c.clearRect(0, 0, aw, ah);
  const r0 = m.cv.getBoundingClientRect(), r1 = a.getBoundingClientRect();
  const ox = (r0.left - r1.left) * dpr, oy = (r0.top - r1.top) * dpr, w = m.cv.width, hh = m.cv.height;
  const ppm = pxPerMetre(m) / dpr;
  if (!(ppm > 0)) return;
  c.fillStyle = "#000"; c.strokeStyle = "#000"; c.lineWidth = 1 * dpr;
  c.font = `${11 * dpr}px Helvetica, Arial, sans-serif`;
  const step = niceStep(60 / ppm);
  // bottom: easting
  const [cx0] = toCell(m, 0, 0), [cx1] = toCell(m, w, 0);
  c.textAlign = "center"; c.textBaseline = "top";
  for (let X = Math.ceil(Math.max(0, cx0 * geo.dx) / step) * step; X <= Math.min(nx * geo.dx, cx1 * geo.dx) + 1e-9; X += step) {
    const sx = toScreen(m, X / geo.dx, 0)[0]; if (sx < -0.5 || sx > w + 0.5) continue;
    c.beginPath(); c.moveTo(ox + sx, oy + hh + dpr); c.lineTo(ox + sx, oy + hh + 5 * dpr); c.stroke();
    c.fillText(String(X), ox + sx, oy + hh + 7 * dpr);
  }
  c.fillText("Easting (m)", ox + w / 2, oy + hh + 19 * dpr);
  // left: northing (increases upward)
  const cy0 = toCell(m, 0, 0)[1], cy1 = toCell(m, 0, hh)[1];
  const Ytop = (ny - cy0) * geo.dy, Ybot = (ny - cy1) * geo.dy;
  c.textAlign = "right"; c.textBaseline = "middle";
  for (let Y = Math.ceil(Math.max(0, Ybot) / step) * step; Y <= Math.min(ny * geo.dy, Ytop) + 1e-9; Y += step) {
    const sy = toScreen(m, 0, ny - Y / geo.dy)[1]; if (sy < -0.5 || sy > hh + 0.5) continue;
    c.beginPath(); c.moveTo(ox - dpr, oy + sy); c.lineTo(ox - 5 * dpr, oy + sy); c.stroke();
    c.fillText(String(Y), ox - 7 * dpr, oy + sy);
  }
  c.save(); c.translate(ox - 38 * dpr, oy + hh / 2); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.textBaseline = "top";
  c.fillText("Northing (m)", 0, 0); c.restore();
}

// Horizontal colorbar drawn from the same ramps as the shader.
function colorbar(m) {
  const dpr = devicePixelRatio, cb = m.cb;
  const W = Math.round(cb.clientWidth * dpr), H = Math.round(cb.clientHeight * dpr);
  if (cb.width !== W || cb.height !== H) { cb.width = W; cb.height = H; }
  const c = cb.getContext("2d"); c.clearRect(0, 0, W, H);
  const x0 = ML * dpr, x1 = W - 18 * dpr, y0 = 4 * dpr, bh = 12 * dpr, bw = x1 - x0;
  const diff = m.mode === 1;
  for (let i = 0; i < bw; i++) { c.fillStyle = rampJS(diff ? DIFF_RAMP : DEPTH_RAMP, (i + 0.5) / bw); c.fillRect(x0 + i, y0, 1, bh); }
  c.strokeStyle = "#000"; c.lineWidth = 1 * dpr; c.strokeRect(x0 - 0.5 * dpr, y0 - 0.5 * dpr, bw + dpr, bh + dpr);
  const ticks = diff ? [-0.2, -0.1, 0, 0.1, 0.2].map(v => [(v / DIFF_MAX + 1) / 2, v === 0 ? "0" : (v < 0 ? MINUS : "") + Math.abs(v)])
                     : [0.005, 0.02, 0.1, 0.5, 2].map(v => [depthT(v), String(v)]);
  c.fillStyle = "#000"; c.font = `${11 * dpr}px Helvetica, Arial, sans-serif`; c.textAlign = "center"; c.textBaseline = "top";
  for (const [t, s] of ticks) {
    const x = x0 + t * bw;
    c.beginPath(); c.moveTo(x, y0 + bh); c.lineTo(x, y0 + bh + 4 * dpr); c.stroke();
    c.fillText(s, x, y0 + bh + 6 * dpr);
  }
  c.fillText(m.cbCaption, x0 + bw / 2, y0 + bh + 21 * dpr);
}

// ---------- state ----------
const maps = [];
const S = { t: 0, layer: "depth" };
const nT = Math.max(...Object.values(frames).map(f => f.length), 1);

function caption() {
  const f = Object.values(frames)[0], met = D.metrics || {};
  const peak = S.layer === "peak", tt = f && f[S.t] ? f[S.t].t.toFixed(0) : "?";
  const what = peak ? "peak water depth over the simulated event" : `water depth at t = ${tt} s`;
  const hasAfter = opt && D.after;
  let s = `Figure 1. Simulated ${what} ` + (hasAfter
    ? `for the no-change terrain (a) and the optimized design (b), and their difference (c).`
    : opt ? `for the no-change terrain.` : `for the input terrain.`);
  s += ` Grid ${nx} × ${ny} cells at ${geo.dx} m. Axes give easting and northing in metres from the south-west corner of the grid.` +
    ` Depth is on a logarithmic scale (cells shallower than ${D_MIN * 1000} mm are left uncoloured); terrain is a grayscale hillshade (vertical exaggeration ×${$("vex").value}).`;
  if (hasAfter && $("worse").checked) s += ` Hatched, outlined cells in (b) are newly worsened (peak depth up by more than 1 cm).`;
  s += ` Black outlines mark protected assets${D.earthworks ? "; dashed dark-red outlines mark earthworks" : ""}.`;
  if (!peak && Object.values(vel).length && $("arrows").checked) s += ` Arrows show unit discharge q (length proportional to √(q/qₕ), qₕ the 95th percentile).`;
  s += ` First-order finite-volume shallow-water model${met.precision || D.precision ? ` (${met.precision || D.precision} arithmetic)` : ""}; features narrower than ~${(3 * geo.dx).toFixed(0)} m are not resolved. Model output, not an engineering approval.`;
  return s;
}

function update() {
  for (const m of maps) {
    if (!m) continue;
    const zKey = m.name === "after" ? "terrain_after" : (R.terrain_before ? "terrain_before" : "terrain");
    if (m._z !== zKey) { m.up(m.T.z, R[zKey] || R.terrain_before || R.terrain); m._z = zKey; }
    let hh;
    if (m.name === "diff") {
      if (S.layer === "peak") hh = R.delta_peak;
      else { const a = frame("before", S.t), b = frame("after", Math.min(S.t, frames.after.length - 1)); hh = new Float32Array(a.length); for (let i = 0; i < a.length; i++) hh[i] = b[i] - a[i]; }
      m.mode = 1; m.cbCaption = S.layer === "peak" ? "Δ peak depth (m), after − before" : "Δ depth at t (m), after − before";
    } else {
      hh = S.layer === "peak" ? (m.name === "after" ? R.peak_after : (R.peak_before)) : frame(m.name, Math.min(S.t, frames[m.name].length - 1));
      m.mode = 0; m.cbCaption = (S.layer === "peak" ? "Peak water depth" : "Water depth") + " (m), log scale";
    }
    m.up(m.T.h, hh);
    draw(m); colorbar(m);
  }
  const f = Object.values(frames)[0];
  $("tlabel").textContent = f && f[S.t] ? `t = ${f[S.t].t.toFixed(0)} s${f[S.t].sat ? " (depth saturated)" : ""}` : "";
  $("caption").textContent = caption();
}

function summaryTable(met) {
  const rows = [], add = (k, v) => rows.push([k, v]);
  const b = met.baseline, cd = met.candidate, pd = met.peak_depth_change, ob = met.objective;
  if (b && cd) {
    add("J before", fmt(b.j, 4)); add("J after", fmt(cd.j, 4));
    add("ΔJ (%)", b.j ? sfmt((cd.j - b.j) / b.j * 100, 1) : "n/a");
    add("Cut (m³)", cd.cut_m3.toFixed(0)); add("Fill (m³)", cd.fill_m3.toFixed(0));
    add("Guard max worsening (m)", fmt(cd.guard_max_worsening_m, 3));
    add("Assets exceeding threshold", `${b.assets_exceeding} → ${cd.assets_exceeding}`);
    if (pd) { add("Area worsened >1 cm (m²)", pd.area_worsened_gt_1cm_m2.toFixed(0)); add("Area improved >1 cm (m²)", pd.area_improved_gt_1cm_m2.toFixed(0)); }
  } else if (ob) {
    add("J", fmt(ob.j, 4)); add("Assets exceeding threshold", String(ob.assets_exceeding));
    if (met.run) { add("Time steps", String(met.run.steps)); add("Simulated time (s)", met.run.t_end.toFixed(0)); }
    if (met.mass_balance) add("Mass residual (relative)", met.mass_balance.rel_residual.toExponential(2));
  }
  add("Grid", `${nx} × ${ny} @ ${geo.dx} m`);
  const prec = met.precision || D.precision; if (prec) add("Precision", String(prec));
  const t = $("summary");
  for (let i = 0; i < rows.length; i += 3) {
    const tr = h("tr");
    for (const [k, v] of rows.slice(i, i + 3)) tr.append(h("th", null, k), h("td", { class: "num" }, v));
    for (let k = Math.min(3, rows.length - i); k < 3; k++) tr.append(h("th"), h("td"));
    t.append(tr);
  }
}

function dataTables(met) {
  const box = $("tables");
  const mk = (cap, head, rows) => {
    const t = h("table"), tr = h("tr");
    head.forEach((s, i) => tr.append(h("th", i ? { class: "num" } : null, s))); t.append(tr);
    for (const r of rows) { const row = h("tr"); r.forEach((s, i) => row.append(h("td", i ? { class: "num" } : null, s))); t.append(row); }
    box.append(h("div", { class: "tbl" }, h("div", { class: "cap" }, cap), t));
  };
  if (met.baseline && met.candidate) {
    mk("Table 1. Protected assets: peak water depth before and after the design.", ["Asset", "Threshold (m)", "Peak before (m)", "Peak after (m)", "Change (m)"],
      met.baseline.assets.map((a, i) => { const c = met.candidate.assets[i]; return [a.name, fmt(a.threshold_m), fmt(a.peak_depth_m), fmt(c.peak_depth_m), sfmt(c.peak_depth_m - a.peak_depth_m)]; }));
  } else if (met.objective) {
    mk("Table 1. Protected assets: peak water depth.", ["Asset", "Threshold (m)", "Peak depth (m)", "Exceedance (m)"],
      met.objective.assets.map(a => [a.name, fmt(a.threshold_m), fmt(a.peak_depth_m), fmt(a.exceedance_m)]));
  }
  if (D.earthworks && D.earthworks.features.length) {
    mk(`Table ${box.children.length + 1}. Earthworks in the optimized design.`, ["#", "Kind", "Length (m)", "Width (m)", "Angle (°)", "Height (m)"],
      D.earthworks.features.map(f => { const p = f.properties; return [String(p.index), p.kind, +(+p.length_m).toFixed(1) + "", +(+p.width_m).toFixed(1) + "", +(+p.angle_deg).toFixed(1) + "", fmt(+p.height_m, 2)]; }));
  }
}

function init() {
  $("maps").className = opt && D.after ? "c3" : "c1";
  if (opt) { maps.push(mkMap("before", "(a) Before (no change)")); if (D.after) { maps.push(mkMap("after", "(b) After (best design)")); maps.push(mkMap("diff", "(c) Difference, after − before")); } }
  else maps.push(mkMap("before", "(a) Simulation"));
  const met = D.metrics || {};
  const sub = [`Scenario kind: ${met.kind || D.kind}`];
  if (met.scenario_id || D.scenario_id) sub.unshift(`Scenario: ${met.scenario_id || D.scenario_id}`);
  if (met.method) sub.push(`method: ${met.method}`);
  if (met.outcome) sub.push(`outcome: ${met.outcome}`);
  if (met.stopped_by) sub.push(`stopped by: ${met.stopped_by}`);
  $("subtitle").textContent = sub.join(" · ");
  summaryTable(met); dataTables(met);
  // legend: only the items that occur in this run
  const lg = $("legend"), item = (cls, s) => lg.append(h("span", null, h("i", { class: cls }), s));
  if ((D.assets || []).length) item("sw-asset", "Protected asset");
  if (D.earthworks) item("sw-earth", "Earthwork");
  if (opt && D.after) item("sw-worse", "Newly worsened (>1 cm)");
  if (Object.keys(vel).length) item("sw-arrow", "Discharge (q)");
  if (!(opt && D.after)) $("worse").parentElement.hidden = true;
  const ts = $("time"); ts.max = nT - 1;
  S.t = Math.floor((nT - 1) / 2); ts.value = S.t;
  ts.oninput = () => { S.t = +ts.value; update(); };
  $("layer").onchange = e => { S.layer = e.target.value; update(); };
  for (const id of ["arrows", "worse", "vex"]) $(id).onchange = update;
  let playing = null;
  $("play").onclick = e => {
    if (playing) { clearInterval(playing); playing = null; e.target.textContent = "Play"; return; }
    e.target.textContent = "Pause";
    playing = setInterval(() => { S.t = (S.t + 1) % nT; ts.value = S.t; update(); }, 120);
  };
  for (const m of maps) {
    if (!m) continue;
    let drag = null;
    m.ov.onmousedown = e => { drag = [e.offsetX, e.offsetY, V.cx, V.cy, false]; };
    window.addEventListener("mouseup", e => { if (drag && !drag[4]) inspect(m, e); drag = null; });
    m.ov.onmousemove = e => {
      if (!drag) return;
      const [a0] = toCell(m, drag[0] * devicePixelRatio, drag[1] * devicePixelRatio), [a1] = toCell(m, e.offsetX * devicePixelRatio, e.offsetY * devicePixelRatio);
      const b0 = toCell(m, drag[0] * devicePixelRatio, drag[1] * devicePixelRatio)[1], b1 = toCell(m, e.offsetX * devicePixelRatio, e.offsetY * devicePixelRatio)[1];
      if (Math.abs(e.offsetX - drag[0]) + Math.abs(e.offsetY - drag[1]) > 3) drag[4] = true;
      V.cx = drag[2] - (a1 - a0) + (V.cx - drag[2]); V.cy = drag[3] - (b1 - b0) + (V.cy - drag[3]);
      drag[0] = e.offsetX; drag[1] = e.offsetY; drag[2] = V.cx; drag[3] = V.cy;
      maps.forEach(x => x && draw(x));
    };
    m.ov.onwheel = e => { e.preventDefault(); V.span *= Math.exp(e.deltaY * 0.001); maps.forEach(x => x && draw(x)); };
    m.ov.ondblclick = () => { V.cx = nx / 2; V.cy = ny / 2; V.span = Math.min(nx, ny); maps.forEach(x => x && draw(x)); };
  }
  window.onresize = () => maps.forEach(m => { if (m) { draw(m); colorbar(m); } });
  update();
}

function inspect(m, e) {
  const r = m.ov.getBoundingClientRect();
  const [x, y] = toCell(m, (e.clientX - r.left) * devicePixelRatio, (e.clientY - r.top) * devicePixelRatio);
  const c = Math.floor(x), w = Math.floor(y);
  if (c < 0 || w < 0 || c >= nx || w >= ny) return;
  const k = w * nx + c, met = D.metrics || {};
  const rows = [];
  rows.push(["Cell (col, row)", `${c}, ${w}`]);
  rows.push(["Easting, northing (m)", `${(geo.origin_x + (c + .5) * geo.dx).toFixed(1)}, ${(geo.origin_y - (w + .5) * geo.dy).toFixed(1)}`]);
  const pb = R.peak_before ? R.peak_before[k] : NaN, pa = R.peak_after ? R.peak_after[k] : NaN;
  rows.push([opt ? "Peak depth before (m)" : "Peak depth (m)", fmt(pb)]);
  if (!isNaN(pa)) { rows.push(["Peak depth after (m)", fmt(pa)]); rows.push(["Change (m)", sfmt(pa - pb)]); }
  (D.assets || []).forEach((a, i) => {
    if (!a.cells.includes(k)) return;
    const rb = met.baseline && met.baseline.assets[i], rc = met.candidate && met.candidate.assets[i];
    const ob = met.objective && met.objective.assets[i];
    const ref = rb || ob;
    rows.push([`Asset ${a.name}`, `threshold ${ref ? fmt(ref.threshold_m) : "?"} m, peak ${ref ? fmt(ref.peak_depth_m) : "?"} m` +
      (rc ? ` → ${fmt(rc.peak_depth_m)} m (change ${sfmt(rc.peak_depth_m - rb.peak_depth_m)} m)` : "")]);
  });
  if (D.earthworks) {
    const wx = geo.origin_x + x * geo.dx, wy = geo.origin_y - y * geo.dy;
    for (const f of D.earthworks.features) {
      const ring = f.geometry.coordinates[0]; let inside = false;
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i], [xj, yj] = ring[j];
        if ((yi > wy) !== (yj > wy) && wx < (xj - xi) * (wy - yi) / (yj - yi) + xi) inside = !inside;
      }
      if (inside) { const p = f.properties; rows.push([`Earthwork #${p.index} (${p.kind})`, `length ${+(+p.length_m).toFixed(1)} m, width ${+(+p.width_m).toFixed(1)} m, angle ${+(+p.angle_deg).toFixed(1)}°, height ${fmt(+p.height_m, 2)} m`]); }
    }
  }
  const t = $("inspect"); t.textContent = "";
  for (const [k2, v] of rows) t.append(h("tr", null, h("th", null, k2), h("td", null, v)));
}

init();
