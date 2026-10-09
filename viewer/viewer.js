// itr static viewer (spec §11): WebGL2, no framework, no build step.
// Three synchronized maps (before / after / difference), time scrubber, discharge
// arrows, asset and earthwork inspection, newly-worsened overlay.
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

// ---------- WebGL ----------
const VS = `#version 300 es
in vec2 p; uniform vec4 view; uniform vec2 grid; out vec2 cell;
void main(){ cell = p * grid; vec2 s = (cell - view.xy) * view.z; gl_Position = vec4(s.x, -s.y * view.w, 0., 1.); }`;
const FS = `#version 300 es
precision highp float; precision highp sampler2D;
in vec2 cell; out vec4 o;
uniform sampler2D z, h, dz; uniform int mode; uniform float vex, cs, worse; uniform vec2 grid;
float Z(ivec2 c){ c = clamp(c, ivec2(0), ivec2(grid) - 1); float v = texelFetch(z, c, 0).r; return isnan(v) ? 1e9 : v; }
vec3 depthCol(float d){ float t = clamp(log(1. + d * 50.) / log(1. + 2. * 50.), 0., 1.);
  return mix(vec3(.75,.9,1.), vec3(.02,.15,.55), t); }
vec3 div(float d){ float t = clamp(d / .2, -1., 1.); return t > 0. ? mix(vec3(.95), vec3(.8,.1,.1), t) : mix(vec3(.95), vec3(.1,.35,.85), -t); }
void main(){
  if (cell.x < 0. || cell.y < 0. || cell.x >= grid.x || cell.y >= grid.y) { o = vec4(.08,.09,.1,1.); return; }
  ivec2 c = ivec2(cell);
  float zc = texelFetch(z, c, 0).r;
  if (isnan(zc) || zc > 1e8) { o = vec4(.25,.25,.27,1.); return; }
  float gx = (Z(c + ivec2(1,0)) - Z(c - ivec2(1,0))) / (2. * cs) * vex;
  float gy = (Z(c + ivec2(0,1)) - Z(c - ivec2(0,1))) / (2. * cs) * vex;
  if (abs(gx) > 1e6) gx = 0.; if (abs(gy) > 1e6) gy = 0.;
  vec3 n = normalize(vec3(-gx, gy, 1.));
  float shade = clamp(dot(n, normalize(vec3(-.5, .5, .7))), 0., 1.);
  vec3 base = vec3(.55,.52,.45) * (.35 + .75 * shade);
  float v = texelFetch(h, c, 0).r;
  vec3 col = base;
  if (mode == 0) { if (v > .005) col = mix(base, depthCol(v), .85); }
  else { col = mix(base, div(v), abs(v) > .005 ? .9 : .0); }
  if (worse > .5) { float d = texelFetch(dz, c, 0).r; if (d > .01 && fract((cell.x + cell.y) * .5) < .5) col = mix(col, vec3(1.,0.,1.), .6); }
  o = vec4(col, 1.);
}`;

function mkMap(name, label) {
  const el = document.createElement("div"); el.className = "map";
  const cv = document.createElement("canvas"), ov = document.createElement("canvas");
  const lb = document.createElement("div"); lb.className = "label"; lb.textContent = label;
  const lg = document.createElement("div"); lg.className = "legend";
  el.append(cv, ov, lb, lg); document.getElementById("maps").append(el);
  const gl = cv.getContext("webgl2");
  if (!gl) { lb.textContent = "WebGL2 unavailable"; return null; }
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
  const m = { name, el, cv, ov, gl, pr, T, up, lg, mode: 0 };
  return m;
}

function draw(m) {
  const { gl, pr, T, cv } = m;
  const w = cv.clientWidth * devicePixelRatio, h = cv.clientHeight * devicePixelRatio;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; m.ov.width = w; m.ov.height = h; }
  gl.viewport(0, 0, w, h); gl.useProgram(pr);
  const sc = 2 / (V.span), aspect = w / h;
  gl.uniform4f(gl.getUniformLocation(pr, "view"), V.cx, V.cy, sc / Math.max(aspect, 1) * (aspect < 1 ? 1 : 1), aspect);
  gl.uniform2f(gl.getUniformLocation(pr, "grid"), nx, ny);
  gl.uniform1i(gl.getUniformLocation(pr, "mode"), m.mode);
  gl.uniform1f(gl.getUniformLocation(pr, "vex"), +document.getElementById("vex").value);
  gl.uniform1f(gl.getUniformLocation(pr, "cs"), geo.dx);
  gl.uniform1f(gl.getUniformLocation(pr, "worse"), m.name === "after" && document.getElementById("worse").checked ? 1 : 0);
  [["z", 0], ["h", 1], ["dz", 2]].forEach(([k, i]) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, T[k]); gl.uniform1i(gl.getUniformLocation(pr, k), i); });
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  overlay(m, sc, aspect);
}

// View: centre (cells) and span (cells across the canvas width).
const V = { cx: nx / 2, cy: ny / 2, span: Math.max(nx, ny) * 1.05 };
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

function overlay(m) {
  const c = m.ov.getContext("2d"); c.clearRect(0, 0, m.ov.width, m.ov.height);
  c.lineWidth = 1.5 * devicePixelRatio;
  // assets
  // assets: outline = edges between member and non-member cells
  c.strokeStyle = "#ffd400"; c.beginPath();
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
  // earthworks
  if (D.earthworks && m.name !== "before") {
    c.strokeStyle = "#ff8c1a";
    for (const f of D.earthworks.features) {
      c.beginPath();
      f.geometry.coordinates[0].forEach(([x, y], i) => { const [cx, cy] = worldToCell(x, y); const [sx, sy] = toScreen(m, cx, cy); i ? c.lineTo(sx, sy) : c.moveTo(sx, sy); });
      c.stroke();
    }
  }
  // arrows
  const run = m.name === "diff" ? null : m.name;
  if (run && vel[run] && document.getElementById("arrows").checked && S.layer === "depth") {
    const v = vel[run], per = v.nx * v.ny * 2, off = S.t * per, st = v.stride;
    c.strokeStyle = "rgba(255,255,255,.85)"; c.lineWidth = 1 * devicePixelRatio;
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
}

// ---------- state ----------
const maps = [];
const S = { t: 0, layer: "depth" };
const nT = Math.max(...Object.values(frames).map(f => f.length), 1);

function update() {
  for (const m of maps) {
    const zKey = m.name === "after" ? "terrain_after" : (R.terrain_before ? "terrain_before" : "terrain");
    if (m._z !== zKey) { m.up(m.T.z, R[zKey] || R.terrain_before || R.terrain); m._z = zKey; }
    let h;
    if (m.name === "diff") {
      if (S.layer === "peak") h = R.delta_peak;
      else { const a = frame("before", S.t), b = frame("after", Math.min(S.t, frames.after.length - 1)); h = new Float32Array(a.length); for (let i = 0; i < a.length; i++) h[i] = b[i] - a[i]; }
      m.mode = 1; m.lg.textContent = "after − before: blue lower, red higher (±0.2 m)";
    } else {
      h = S.layer === "peak" ? (m.name === "after" ? R.peak_after : (R.peak_before)) : frame(m.name, Math.min(S.t, frames[m.name].length - 1));
      m.mode = 0; m.lg.textContent = "depth (log scale, 5 mm – 2 m)";
    }
    m.up(m.T.h, h);
    draw(m);
  }
  const f = Object.values(frames)[0];
  document.getElementById("tlabel").textContent = f && f[S.t] ? `t = ${f[S.t].t.toFixed(0)} s${f[S.t].sat ? " (depth saturated in frame)" : ""}` : "";
}

function init() {
  document.getElementById("maps").style.setProperty("--cols", opt ? 3 : 1);
  if (opt) { maps.push(mkMap("before", "Before (no change)")); if (D.after) { maps.push(mkMap("after", "After (best design)")); maps.push(mkMap("diff", "Difference")); } }
  else maps.push(mkMap("before", "Simulation"));
  const met = D.metrics || {};
  document.getElementById("title").textContent = `itr viewer — ${met.kind || D.kind}${met.outcome ? " — " + met.outcome : ""}`;
  const parts = [];
  if (met.baseline) parts.push(`J before ${met.baseline.j.toFixed(4)}`);
  if (met.candidate) parts.push(`J after ${met.candidate.j.toFixed(4)}`, `cut ${met.candidate.cut_m3.toFixed(0)} m³`, `fill ${met.candidate.fill_m3.toFixed(0)} m³`,
    `guard worsening ${met.candidate.guard_max_worsening_m.toFixed(3)} m`);
  document.getElementById("summary").textContent = parts.join(" · ");
  document.getElementById("warn").textContent =
    `Model output on a ${geo.dx} m grid (first-order SWE); features narrower than ~${(3 * geo.dx).toFixed(0)} m are not resolved. Not an engineering approval.`;
  const ts = document.getElementById("time"); ts.max = nT - 1;
  S.t = Math.floor((nT - 1) / 2); ts.value = S.t;
  ts.oninput = () => { S.t = +ts.value; update(); };
  document.getElementById("layer").onchange = e => { S.layer = e.target.value; update(); };
  for (const id of ["arrows", "worse", "vex"]) document.getElementById(id).onchange = update;
  let playing = null;
  document.getElementById("play").onclick = e => {
    if (playing) { clearInterval(playing); playing = null; e.target.textContent = "▶"; return; }
    e.target.textContent = "⏸";
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
      maps.forEach(draw);
    };
    m.ov.onwheel = e => { e.preventDefault(); V.span *= Math.exp(e.deltaY * 0.001); maps.forEach(draw); };
  }
  window.onresize = () => maps.forEach(draw);
  update();
}

function inspect(m, e) {
  const r = m.ov.getBoundingClientRect();
  const [x, y] = toCell(m, (e.clientX - r.left) * devicePixelRatio, (e.clientY - r.top) * devicePixelRatio);
  const c = Math.floor(x), w = Math.floor(y);
  if (c < 0 || w < 0 || c >= nx || w >= ny) return;
  const k = w * nx + c, met = D.metrics || {};
  const lines = [`cell (${c}, ${w})  x ${(geo.origin_x + (c + .5) * geo.dx).toFixed(1)}  y ${(geo.origin_y - (w + .5) * geo.dy).toFixed(1)}`];
  const pb = R.peak_before ? R.peak_before[k] : NaN, pa = R.peak_after ? R.peak_after[k] : NaN;
  lines.push(`peak depth before ${pb.toFixed(3)} m` + (isNaN(pa) ? "" : `  after ${pa.toFixed(3)} m  change ${(pa - pb >= 0 ? "+" : "") + (pa - pb).toFixed(3)} m`));
  (D.assets || []).forEach((a, i) => {
    if (!a.cells.includes(k)) return;
    const rb = met.baseline && met.baseline.assets[i], rc = met.candidate && met.candidate.assets[i];
    const ob = met.objective && met.objective.assets[i];
    const ref = rb || ob;
    lines.push(`asset ${a.name}: threshold ${ref ? ref.threshold_m.toFixed(3) : "?"} m, peak ${ref ? ref.peak_depth_m.toFixed(3) : "?"} m` +
      (rc ? ` → ${rc.peak_depth_m.toFixed(3)} m (change ${(rc.peak_depth_m - rb.peak_depth_m).toFixed(3)} m)` : ""));
  });
  if (D.earthworks) {
    const wx = geo.origin_x + x * geo.dx, wy = geo.origin_y - y * geo.dy;
    for (const f of D.earthworks.features) {
      const ring = f.geometry.coordinates[0]; let inside = false;
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i], [xj, yj] = ring[j];
        if ((yi > wy) !== (yj > wy) && wx < (xj - xi) * (wy - yi) / (yj - yi) + xi) inside = !inside;
      }
      if (inside) { const p = f.properties; lines.push(`earthwork #${p.index} ${p.kind}: length ${p.length_m} m, width ${p.width_m} m, angle ${p.angle_deg}°, height ${p.height_m} m`); }
    }
  }
  document.getElementById("info").textContent = lines.join("\n");
}

init();
