#!/usr/bin/env python3
"""3D simulation video of an `itr optimize` run (docs/video/<example>-3d.mp4).

Two linked 3D views of the terrain, no change and optimized design, with the recorded
water surface, rain, protected buildings that turn red when flooded above their
threshold, and the downstream guard area outlined. The camera orbits slowly.

Display only: heights are exaggerated (the factor is printed on the video), the terrain
and depth grids are upsampled for a smooth surface, and depth frames between sync
intervals are linearly interpolated. Building depths are read from the simulation
grid, not the upsampled one.

Not part of the engine; needs numpy, scipy, pyvista, pillow, tifffile, lz4, matplotlib
and ffmpeg on PATH:

    uv run --with numpy --with scipy --with pyvista --with pillow --with tifffile --with lz4 --with matplotlib \\
        scripts/simulation_video_3d.py --example rolling-hills
"""

import argparse
import json
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import zoom

from figures import ROOT, polys, read_tif
from simulation_video import lerp, read_frames

W, H = 1920, 1080
SANS = "/System/Library/Fonts/Supplemental/Arial.ttf"
SANS_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"
WET = 0.005
SKY_TOP, SKY_BOT = "#0b1422", "#3d5468"
FLOOD, SAFE, ACCENT = (235, 72, 54), (98, 200, 120), (240, 150, 60)


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def card(img, box, radius=14, alpha=150):
    """Translucent dark rounded rectangle composited onto img (RGBA)."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(box, radius, fill=(8, 14, 24, alpha))
    img.alpha_composite(layer)


def house_mesh(x0, x1, y0, y1, zg, wall_h, roof_h):
    """Walls (box) and a gabled roof (triangular prism along the longer side)."""
    walls = pv.Box((x0, x1, y0, y1, zg - 2.0, zg + wall_h))
    o = 0.6  # eave overhang
    zt, zr = zg + wall_h, zg + wall_h + roof_h
    if x1 - x0 >= y1 - y0:
        ym = (y0 + y1) / 2
        pts = [(x0 - o, y0 - o, zt), (x1 + o, y0 - o, zt), (x1 + o, ym, zr), (x0 - o, ym, zr),
               (x0 - o, y1 + o, zt), (x1 + o, y1 + o, zt)]
        faces = [4, 0, 1, 2, 3, 4, 3, 2, 5, 4, 3, 0, 3, 4, 3, 1, 5, 2]
    else:
        xm = (x0 + x1) / 2
        pts = [(x0 - o, y0 - o, zt), (x0 - o, y1 + o, zt), (xm, y1 + o, zr), (xm, y0 - o, zr),
               (x1 + o, y0 - o, zt), (x1 + o, y1 + o, zt)]
        faces = [4, 0, 1, 2, 3, 4, 3, 2, 5, 4, 3, 0, 3, 4, 3, 1, 5, 2]
    return walls, pv.PolyData(np.array(pts, float), faces=faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="diverted-flood")
    ap.add_argument("--run", type=Path, help="itr optimize output (default runs/readme/<example>)")
    ap.add_argument("--out", type=Path, help="default docs/video/<example>-3d.mp4")
    ap.add_argument("--seconds", type=float, default=20.0, help="duration of the simulated-time sweep")
    ap.add_argument("--hold", type=float, default=3.0, help="still frames at the end (s)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--upsample", type=int, default=3, help="display-mesh refinement of the simulation grid")
    ap.add_argument("--vert-exag", type=float, help="default: relief scaled to ~12%% of the domain width")
    ap.add_argument("--shadows", action="store_true", help="VTK shadow pass (can break the per-frame water mesh)")
    args = ap.parse_args()
    run = args.run or ROOT / "runs" / "readme" / args.example
    out = args.out or ROOT / "docs" / "video" / f"{args.example}-3d.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    scen_dir = ROOT / "examples" / args.example

    idx = json.loads((run / "viewer_index.json").read_text())
    scen = tomllib.loads((scen_dir / "scenario.toml").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    ts, hb, _, _ = read_frames(run, "before", idx)
    _, ha, _, _ = read_frames(run, "after", idx)
    ny, nx = idx["ny"], idx["nx"]
    zb, ext, origin = read_tif(run / "terrain_before.tif")
    za, _, _ = read_tif(run / "terrain_after.tif")
    dz, _, _ = read_tif(run / "terrain_delta.tif")
    zfill = np.nanmin(zb)
    zb, za, dz = np.nan_to_num(zb, nan=zfill), np.nan_to_num(za, nan=zfill), np.nan_to_num(dz)
    cell = (ext[1] - ext[0]) / nx
    width = max(ext[1], ext[3])
    z0 = zb.min()
    ve = args.vert_exag or float(np.clip(0.12 * width / max(np.ptp(zb), 1e-6), 1, 20))

    # Display mesh: cell-centre grid refined by `up`, rows north-first, Fortran point order.
    up = max(1, args.upsample)
    mx, my = nx * up, ny * up
    xs = (np.arange(mx) + 0.5) * (ext[1] / mx)
    ys = ext[3] - (np.arange(my) + 0.5) * (ext[3] / my)
    X, Y = np.meshgrid(xs, ys)
    F = lambda a: np.asarray(a).ravel(order="F")  # noqa: E731

    def refine(a, order):
        return zoom(a, up, order=order, mode="nearest", grid_mode=True) if up > 1 else a

    def zs(z):
        return (z - z0) * ve

    land = LinearSegmentedColormap.from_list("land", ["#3f6b35", "#5f8a3f", "#8ea455", "#b9b06c", "#cdbf8f", "#e2d9bd"])
    water = LinearSegmentedColormap.from_list("water", ["#9fe3f5", "#45b4e0", "#1f7fc0", "#0d4e94", "#082d63"])
    rng = np.random.default_rng(0)
    speckle = refine(rng.standard_normal((ny, nx)), 1) * 0.035 + rng.standard_normal((my, mx)) * 0.02

    def terrain_rgb(z, d):
        c = land((z - zb.min()) / max(np.ptp(zb), 1e-6))[..., :3]
        gy, gx = np.gradient(z, ext[1] / mx)
        steep = np.clip(np.hypot(gx, gy) / 0.15, 0, 1)[..., None]
        c = c * (1 - 0.5 * steep) + np.array([0.55, 0.45, 0.32]) * 0.5 * steep
        c = np.clip(c * (1 + speckle[..., None]), 0, 1)
        fill = np.clip(d / 0.25, 0, 1)[..., None]
        cut = np.clip(-d / 0.25, 0, 1)[..., None]
        c = c * (1 - fill) + np.array(ACCENT) / 255 * fill
        c = c * (1 - cut) + np.array([0.42, 0.28, 0.18]) * cut
        return (c.transpose(1, 0, 2).reshape(-1, 3) * 255).astype(np.uint8)

    assets = polys(scen_dir / "protected.geojson", origin)
    guards = polys(scen_dir / "guard.geojson", origin)
    asset_cells = [np.array(a["cells"]) for a in idx["assets"]]
    names = [a["name"] for a in idx["assets"]]
    thresholds = [a["threshold_m"] for a in idx["assets"]]
    hyeto = np.array(scen["hydrology"]["rainfall_hyetograph"], float)
    rain_max = hyeto[:, 1].max()

    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(W, H), border=False, lighting="none")
    pl.set_background(SKY_BOT, top=SKY_TOP)
    n_drops = 700
    drop_xy = rng.uniform([0, 0], [ext[1], ext[3]], (n_drops, 2))
    drop_phase = rng.uniform(0, 1, n_drops)
    sky = zs(zb.max()) + 0.3 * width
    cx, cy = ext[1] / 2, ext[3] / 2
    views = []
    for k, (z, h, d) in enumerate(((zb, hb, np.zeros_like(dz)), (za, ha, dz))):
        pl.subplot(0, k)
        sun = pv.Light(position=(cx - 1.2 * width, cy - 1.0 * width, 1.1 * width), focal_point=(cx, cy, 0), intensity=0.62,
                       color="#fff1dc", light_type="scene light")
        fill_light = pv.Light(position=(cx + width, cy + width, 0.8 * width), focal_point=(cx, cy, 0), intensity=0.22,
                              color="#a9c4e8", light_type="scene light")
        pl.add_light(sun)
        pl.add_light(fill_light)
        zr = refine(z, 3)
        dr_ = refine(d, 1)
        ground = pv.StructuredGrid(X, Y, zs(zr))
        ground.point_data["rgb"] = terrain_rgb(zr, dr_)
        pl.add_mesh(ground, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.18, diffuse=0.9, specular=0.03)
        step = float(np.select([np.ptp(zb) > 20, np.ptp(zb) > 8, np.ptp(zb) > 3], [2.0, 1.0, 0.5], 0.25))
        ground.point_data["elev"] = F(zr)
        iso = ground.contour(isosurfaces=np.arange(np.ceil(zr.min() / step) * step, zr.max(), step), scalars="elev")
        if iso.n_points:
            iso.points = iso.points + [0, 0, 0.003 * width]
            pl.add_mesh(iso, color="#1e2a14", line_width=1.0, opacity=0.22, lighting=False)
        # Sides of the block: layered soil.
        base = -0.06 * width
        for edge in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
            ex, ey, ez = X[edge], Y[edge], zs(zr)[edge]
            skirt = pv.StructuredGrid(np.stack([ex, ex, ex]), np.stack([ey, ey, ey]),
                                      np.stack([ez, ez - 0.012 * width, np.full_like(ez, base)]))
            skirt.point_data["rgb"] = np.tile(np.array([[96, 120, 60], [120, 92, 62], [74, 56, 42]], np.uint8), (len(ex), 1))
            pl.add_mesh(skirt, scalars="rgb", rgb=True, ambient=0.35, diffuse=0.7)
        # Water surface: moved every frame; normals recomputed by hand (smooth_shading=True
        # would render a copy that never sees the updates).
        sea = pv.StructuredGrid(X, Y, zs(zr) - 1.0).extract_surface(pass_pointid=True, algorithm="dataset_surface")
        sea_ids = np.asarray(sea.point_data["vtkOriginalPointIds"])
        sea.point_data["depth"] = np.zeros(sea.n_points)
        pl.add_mesh(sea, scalars="depth", cmap=water, clim=(np.log10(WET), np.log10(0.6)), specular=1.0, specular_power=70,
                    opacity=0.88, show_scalar_bar=False, ambient=0.25, diffuse=0.75, interpolate_before_map=True)
        for x, y in guards:
            xx = np.concatenate([np.linspace(a, b, 40) for a, b in zip(x[:-1], x[1:])])
            yy = np.concatenate([np.linspace(a, b, 40) for a, b in zip(y[:-1], y[1:])])
            zi = zs(zr)[np.clip(((ext[3] - yy) / ext[3] * my).astype(int), 0, my - 1), np.clip((xx / ext[1] * mx).astype(int), 0, mx - 1)]
            pl.add_mesh(pv.lines_from_points(np.c_[xx, yy, zi + 0.008 * width]), color="#ffb347", line_width=4, lighting=False)
        houses = []
        for x, y in assets:
            r = int(np.clip((ext[3] - y.mean()) / cell, 0, ny - 1))
            c = int(np.clip(x.mean() / cell, 0, nx - 1))
            walls, roof = house_mesh(x.min(), x.max(), y.min(), y.max(), zs(z)[r, c], 0.022 * width, 0.014 * width)
            houses.append(pl.add_mesh(walls, color="#efe6d4", ambient=0.3, diffuse=0.8))
            pl.add_mesh(roof, color="#4a4f57", ambient=0.25, diffuse=0.8, specular=0.2)
        drops = pv.PolyData(np.zeros((2 * n_drops, 3)), lines=np.c_[np.full(n_drops, 2), np.arange(0, 2 * n_drops, 2), np.arange(1, 2 * n_drops, 2)].ravel())
        pl.add_mesh(drops, color="#cfe3ff", line_width=1.5, opacity=0.45, lighting=False)
        views.append({"z": zs(zr), "h": h, "sea": sea, "sea_ids": sea_ids, "houses": houses, "drops": drops})
        if args.shadows:
            pl.enable_shadows()
    pl.link_views()
    pl.enable_anti_aliasing("ssaa", all_renderers=True)

    fz = zs(zb).mean()

    def camera(u):
        az = np.deg2rad(-30 + 42 * u)  # orbit, measured from looking north
        dist, elev = 2.75 * width, np.deg2rad(36)
        px = cx + dist * np.cos(elev) * np.sin(az)
        py = cy - dist * np.cos(elev) * np.cos(az)
        return [(px, py, fz + dist * np.sin(elev)), (cx, cy - 0.04 * width, fz), (0, 0, 1)]

    f_title, f_sub, f_clock = font(SANS_B, 32), font(SANS, 19), font(MONO, 34)
    f_panel, f_row, f_small, f_foot = font(SANS_B, 26), font(SANS, 18), font(SANS, 16), font(SANS, 17)
    b, a = metrics["baseline"], metrics.get("candidate") or {}

    n_sweep = int(round(args.seconds * args.fps))
    n_hold = int(round(args.hold * args.fps))
    times = np.concatenate([np.linspace(0, ts[-1], n_sweep), np.full(n_hold, ts[-1])])
    us = np.concatenate([np.linspace(0, 1, n_sweep), 1 + np.linspace(0, 0.06, n_hold)])
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps),
                           "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)
    peak = [[0.0] * len(names) for _ in views]
    dmax_scale = max(max(thresholds) * 3, max(float(hb.reshape(len(ts), -1)[:, c].max()) for c in asset_cells) * 1.1)
    for fi, (t, u) in enumerate(zip(times, us)):
        rain = float(np.interp(t, hyeto[:, 0], hyeto[:, 1], right=0.0))
        status = []
        for vi, v in enumerate(views):
            h = lerp(t, ts, v["h"])
            hr = np.maximum(refine(h, 1), 0)
            wet = hr > WET
            sea, ids = v["sea"], v["sea_ids"]
            pts = sea.points.copy()
            pts[:, 2] = F(np.where(wet, v["z"] + hr * ve, v["z"] - 0.01 * width))[ids]
            sea.points = pts
            sea.point_data["depth"] = F(np.log10(np.maximum(hr, WET)))[ids]
            sea.point_data["Normals"] = sea.compute_normals(cell_normals=False, split_vertices=False)["Normals"]
            sea.point_data.active_normals_name = "Normals"
            row = []
            for ai, cells in enumerate(asset_cells):
                dmax = float(h.ravel()[cells].max())
                peak[vi][ai] = max(peak[vi][ai], dmax)
                over = dmax > thresholds[ai]
                v["houses"][ai].prop.color = "#e8452f" if over else "#efe6d4"
                row.append((names[ai], dmax, over))
            status.append(row)
            # Rain streaks: count follows intensity, positions fall at a fixed speed.
            m = int(round(n_drops * rain / rain_max))
            fall = (drop_phase + fi * 0.04) % 1.0
            top = np.c_[drop_xy, sky * (1 - fall)]
            bot = top - [0.003 * width, 0.0015 * width, 0.045 * width]
            dpts = np.empty((2 * n_drops, 3))
            dpts[0::2], dpts[1::2] = top, bot
            dpts[2 * m:] = [cx, cy, -0.5 * width]  # unused drops hidden under the ground
            v["drops"].points = dpts
        pl.camera_position = camera(u)
        pl.render()
        img = Image.fromarray(pl.screenshot(return_img=True)[..., :3]).convert("RGBA")

        # Header band.
        band = Image.new("RGBA", (W, 120), (0, 0, 0, 0))
        bd = ImageDraw.Draw(band)
        for yy in range(120):
            bd.line((0, yy, W, yy), fill=(5, 10, 18, int(200 * (1 - yy / 120) ** 1.5)))
        img.alpha_composite(band)
        dr = ImageDraw.Draw(img)
        dr.text((44, 26), f"{args.example.replace('-', ' ').title()}: flood simulation with and without earthworks", font=f_title, fill="white")
        dr.text((44, 70), f"2D shallow-water model · {ts[-1] / 60:.0f} min storm · {nx} × {ny} cells at {cell:g} m · "
                          f"heights exaggerated ×{ve:.0f}", font=f_sub, fill=(200, 212, 225))
        mm, ss = divmod(int(round(t)), 60)
        dr.text((W - 44, 22), f"{mm:02d}:{ss:02d}", font=f_clock, fill="white", anchor="ra")
        bx0, bx1, by = W - 330, W - 44, 76
        dr.rounded_rectangle((bx0, by, bx1, by + 12), 6, fill=(58, 68, 82))
        if rain > 0:
            dr.rounded_rectangle((bx0, by, bx0 + max(12, (bx1 - bx0) * rain / rain_max), by + 12), 6, fill=(120, 190, 255))
        dr.text((bx0 - 12, by + 6), f"rain {rain:3.0f} mm/h", font=f_sub, fill=(200, 212, 225), anchor="rm")
        dr.line((W // 2, 130, W // 2, H - 80), fill=(90, 105, 125), width=2)

        for vi, (label, x0) in enumerate((("NO CHANGE", 0), ("WITH OPTIMIZED EARTHWORKS", W // 2))):
            # Panel label pill.
            tw = dr.textlength(label, font=f_panel)
            card(img, (x0 + 36, 138, x0 + 36 + tw + 36, 182), radius=22, alpha=170)
            dr = ImageDraw.Draw(img)
            dr.text((x0 + 54, 160), label, font=f_panel, fill="white" if vi == 0 else (255, 200, 140), anchor="lm")
            # Building status card with depth bars against the limit.
            n = len(status[vi])
            ch = 46 + 34 * n
            cy0 = H - 96 - ch
            card(img, (x0 + 36, cy0, x0 + 440, cy0 + ch))
            dr = ImageDraw.Draw(img)
            dr.text((x0 + 56, cy0 + 14), "Water at buildings", font=f_small, fill=(170, 185, 200))
            for ai, (name, dmax, over) in enumerate(status[vi]):
                yy = cy0 + 46 + 34 * ai
                col = FLOOD if over else SAFE
                dr.ellipse((x0 + 56, yy + 4, x0 + 68, yy + 16), fill=col)
                dr.text((x0 + 78, yy + 10), name.replace("_", " "), font=f_row, fill="white", anchor="lm")
                gx0, gx1 = x0 + 190, x0 + 330
                dr.rounded_rectangle((gx0, yy + 4, gx1, yy + 16), 6, fill=(58, 68, 82))
                fw = (gx1 - gx0) * min(dmax / dmax_scale, 1)
                if fw > 2:
                    dr.rounded_rectangle((gx0, yy + 4, gx0 + fw, yy + 16), 6, fill=col)
                lx = gx0 + (gx1 - gx0) * thresholds[ai] / dmax_scale
                dr.line((lx, yy, lx, yy + 20), fill="white", width=2)
                pk = peak[vi][ai]
                px_ = gx0 + (gx1 - gx0) * min(pk / dmax_scale, 1)
                dr.line((px_, yy + 2, px_, yy + 18), fill=(150, 160, 175), width=1)
                dr.text((x0 + 420, yy + 10), f"{dmax * 100:4.1f} cm", font=f_row, fill=col, anchor="rm")
        dr = ImageDraw.Draw(img)
        if a and fi >= n_sweep:
            saved = sum(not over for _, _, over in status[1])
            gw, tol = a.get("guard_max_worsening_m", 0.0), scen["objectives"]["guard_tolerance_m"]
            msg = (f"{saved} of {len(names)} buildings below their limit  ·  objective {(1 - a['j'] / b['j']) * 100:.0f}% lower  ·  "
                   f"downstream area {gw * 100:+.1f} cm (allowed {tol * 100:g} cm)")
            tw = dr.textlength(msg, font=f_panel)
            card(img, (W / 2 - tw / 2 - 30, 196, W / 2 + tw / 2 + 30, 256), radius=30, alpha=200)
            dr = ImageDraw.Draw(img)
            dr.text((W / 2, 226), msg, font=f_panel, fill="white", anchor="mm")

        # Legend strip.
        card(img, (0, H - 62, W, H), radius=0, alpha=170)
        dr = ImageDraw.Draw(img)
        items = [((69, 160, 220), "water (darker = deeper)"), (ACCENT, "earthwork (added fill)"),
                 ((255, 179, 71), "downstream area that must not get worse"), (FLOOD, "building flooded above its limit"),
                 (None, "flood limit (white tick on the bars)")]
        total = sum(dr.textlength(t, font=f_foot) + 60 for _, t in items)
        xx = (W - total) / 2
        for col, text in items:
            if col is None:
                dr.line((xx + 11, H - 41, xx + 11, H - 21), fill="white", width=2)
            else:
                dr.rounded_rectangle((xx, H - 39, xx + 22, H - 23), 4, fill=col)
            dr.text((xx + 30, H - 31), text, font=f_foot, fill=(215, 225, 235), anchor="lm")
            xx += dr.textlength(text, font=f_foot) + 60
        ff.stdin.write(img.convert("RGB").tobytes())
    ff.stdin.close()
    ff.wait()
    pl.close()
    print(out)


if __name__ == "__main__":
    main()
