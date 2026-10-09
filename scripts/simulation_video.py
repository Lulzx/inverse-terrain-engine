#!/usr/bin/env python3
"""Simulation video of an `itr optimize` run (docs/video/<example>.mp4).

Animates the recorded depth frames of the no-change and optimized runs side by side,
with their difference, the rainfall hyetograph and the depth at each protected asset.
Frames between sync intervals are linearly interpolated for smooth playback.

Frames are rendered in parallel processes (`--jobs`) and encoded with the VideoToolbox
hardware encoder when available (`--encoder`); see videoio.py.

Not part of the engine; needs numpy, matplotlib, tifffile, lz4 and ffmpeg on PATH:

    uv run --with numpy --with matplotlib --with tifffile --with lz4 scripts/simulation_video.py
"""

import argparse
import json
import tomllib
from pathlib import Path

import lz4.block
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, LogNorm, TwoSlopeNorm  # noqa: E402

import videoio  # noqa: E402
from figures import ROOT, polys, read_tif, scalebar  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "figure.facecolor": "white",
})

ASSET_COLORS = ["#1f4e9c", "#b2182b", "#2d7d46", "#7a4fa0"]


def read_frames(run, tag, idx):
    meta = idx[tag]
    ny, nx = idx["ny"], idx["nx"]
    t = np.array([0.0] + [f["t"] for f in meta["frames"]])
    h = [np.zeros((ny, nx))]
    for f in meta["frames"]:
        raw = lz4.block.decompress((run / "frames" / f["file"]).read_bytes())
        h.append(np.frombuffer(raw, "<u2").reshape(ny, nx) / 1000.0)
    v = meta["velocity"]
    q = np.frombuffer(lz4.block.decompress((run / "frames" / v["file"]).read_bytes()), "<f4")
    q = q.reshape(len(meta["frames"]), v["ny"], v["nx"], 2)
    q = np.concatenate([np.zeros_like(q[:1]), q])
    return t, np.array(h), q, v["stride"]


def lerp(t, ts, arr):
    k = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2))
    w = np.clip((t - ts[k]) / (ts[k + 1] - ts[k]), 0, 1)
    return (1 - w) * arr[k] + w * arr[k + 1]


def frame_times(args, t_end):
    n_sweep = int(round(args.seconds * args.fps))
    n_hold = int(round(args.hold * args.fps))
    return np.concatenate([np.linspace(0, t_end, n_sweep), np.full(n_hold, t_end)])


def build(args):
    """Set up the figure; returns draw(i) -> RGBA bytes of frame i (depends on i only)."""
    run = args.run or ROOT / "runs" / "readme" / args.example
    scen_dir = ROOT / "examples" / args.example

    idx = json.loads((run / "viewer_index.json").read_text())
    scen = tomllib.loads((scen_dir / "scenario.toml").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    tb, hb, qb, stride = read_frames(run, "before", idx)
    ta, ha, qa, _ = read_frames(run, "after", idx)
    assert np.array_equal(tb, ta)
    ts = tb
    ny, nx = idx["ny"], idx["nx"]

    zb, ext, origin = read_tif(run / "terrain_before.tif")
    za, _, _ = read_tif(run / "terrain_after.tif")
    cell = (ext[1] - ext[0]) / nx
    ls = LightSource(azdeg=315, altdeg=45)
    zr = (np.nanmin(zb), np.nanmax(zb))
    earth = matplotlib.colors.LinearSegmentedColormap.from_list("earth", ["#c9c3a8", "#e9e4d3", "#f6f3ea"])
    shade = [ls.shade(np.nan_to_num(z, nan=zr[1]), cmap=earth, vmin=zr[0], vmax=zr[1], vert_exag=6, dx=cell, dy=cell, blend_mode="soft") for z in (zb, za, za)]
    water = matplotlib.colors.LinearSegmentedColormap.from_list("water", plt.cm.Blues(np.linspace(0.3, 1.0, 256)))
    assets = polys(scen_dir / "protected.geojson", origin)
    guards = polys(scen_dir / "guard.geojson", origin)
    works = polys(run / "earthworks.geojson", origin)

    # Max depth over each asset footprint, per recorded frame.
    series = []
    for a in idx["assets"]:
        cells = np.array(a["cells"])
        series.append((a["name"], a["threshold_m"], hb.reshape(len(ts), -1)[:, cells].max(1), ha.reshape(len(ts), -1)[:, cells].max(1)))

    hyeto = np.array(scen["hydrology"]["rainfall_hyetograph"], float)
    t_end = ts[-1]
    tr = np.linspace(0, t_end, 400)
    rain = np.interp(tr, hyeto[:, 0], hyeto[:, 1], right=0.0)

    # Arrow grid (velocity samples sit at cell centres stride/2, stride/2 + stride, ...; rows north-first).
    sub = 2
    vny, vnx = qb.shape[1:3]
    cx = (np.arange(vnx) * stride + stride // 2 + 0.5) * cell
    cy = ext[3] - (np.arange(vny) * stride + stride // 2 + 0.5) * cell
    QX, QY = np.meshgrid(cx[::sub], cy[::sub])

    W, H, dpi = 1920, 1080, 120
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 1], left=0.05, right=0.985, top=0.9, bottom=0.07, hspace=0.3, wspace=0.08)
    axs = [fig.add_subplot(gs[0, k]) for k in range(3)]
    ax_rain = fig.add_subplot(gs[1, 0])
    ax_dep = fig.add_subplot(gs[1, 1:])

    b, a = metrics["baseline"], metrics.get("candidate") or {}
    fig.text(0.05, 0.955, f"{args.example}: flood simulation, no change vs optimized earthworks", fontsize=19, weight="bold", va="center")
    clock = fig.text(0.985, 0.955, "", fontsize=19, ha="right", va="center", family="monospace")
    fig.text(0.05, 0.918, f"{nx} × {ny} cells at {cell:g} m · 2D shallow-water equations (HLL, hydrostatic reconstruction, Manning n = {scen['hydrology']['manning_n']}) · "
             f"{t_end / 60:.0f} min storm", fontsize=11.5, color="0.3", va="center")

    norm = LogNorm(0.005, 1.0)
    lim = 0.15
    depth_ims, quivers = [], []
    titles = ["Water depth, no change", "Water depth, optimized design", "Difference (design − no change)"]
    for k, ax in enumerate(axs):
        ax.imshow(shade[k], extent=ext)
        if k < 2:
            im = ax.imshow(np.ma.masked_all((ny, nx)), cmap=water, norm=norm, extent=ext, alpha=0.92, interpolation="bilinear")
            quivers.append(ax.quiver(QX, QY, np.zeros_like(QX), np.zeros_like(QX), color="#0b1f3a", alpha=0.7, scale=1.0, scale_units="xy",
                                     angles="xy", width=0.0032, headwidth=3.5, headlength=3.5, headaxislength=3, pivot="mid", zorder=4))
        else:
            im = ax.imshow(np.ma.masked_all((ny, nx)), cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), extent=ext, interpolation="bilinear")
        depth_ims.append(im)
        for x, y in guards:
            ax.plot(x, y, color="#e08214", lw=1.1, ls=(0, (2, 1.5)), zorder=5)
        for x, y in assets:
            ax.plot(x, y, color="k", lw=1.3, zorder=5)
        if k:
            for x, y in works:
                ax.plot(x, y, color="#b2182b" if k == 1 else "k", lw=1.4, ls="--", zorder=5)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[k], pad=6)
        scalebar(ax, ext, 100)
    cb = fig.colorbar(depth_ims[0], ax=axs[:2], orientation="horizontal", fraction=0.045, pad=0.025, aspect=50)
    cb.set_label("Water depth (m), log scale", fontsize=11)
    cb2 = fig.colorbar(depth_ims[2], ax=axs[2], orientation="horizontal", fraction=0.045, pad=0.025, aspect=25, extend="both")
    cb2.set_label("Δ depth (m)  (red: deeper with design)", fontsize=11)
    # Legend for the map overlays.
    from matplotlib.lines import Line2D
    axs[2].legend(handles=[Line2D([], [], color="k", lw=1.3, label="protected asset"),
                           Line2D([], [], color="#e08214", lw=1.1, ls=(0, (2, 1.5)), label="downstream guard"),
                           Line2D([], [], color="k", lw=1.4, ls="--", label="earthwork")],
                  loc="upper right", fontsize=9.5, frameon=True, framealpha=0.85, edgecolor="none")

    ax_rain.fill_between(tr / 60, rain, color="0.8", lw=0)
    ax_rain.plot(tr / 60, rain, color="0.35", lw=1)
    rain_cur = ax_rain.axvline(0, color="k", lw=0.8)
    ax_rain.set_xlim(0, t_end / 60)
    ax_rain.set_ylim(0, rain.max() * 1.15)
    ax_rain.set_xlabel("Time (min)")
    ax_rain.set_ylabel("Rainfall (mm/h)")
    ax_rain.set_title("Design storm", pad=6)

    lines = []
    dmax = max(max(s[2].max(), s[3].max(), s[1]) for s in series)
    for k, (name, thr, sb, sa) in enumerate(series):
        c = ASSET_COLORS[k % len(ASSET_COLORS)]
        ax_dep.plot(ts / 60, sb, color=c, lw=0.8, ls=":", alpha=0.25)
        ax_dep.plot(ts / 60, sa, color=c, lw=0.8, alpha=0.25)
        lb, = ax_dep.plot([], [], color=c, lw=1.6, ls=":", label=f"{name}, no change (peak {sb.max():.3f} m)")
        la, = ax_dep.plot([], [], color=c, lw=2.2, label=f"{name}, design (peak {sa.max():.3f} m)")
        lines.append((lb, la, sb, sa))
    thr = series[0][1]
    ax_dep.axhline(thr, color="k", lw=0.8, ls=(0, (4, 3)))
    ax_dep.text(t_end / 60 * 0.995, thr, f"threshold {thr:g} m", ha="right", va="bottom", fontsize=10)
    dep_cur = ax_dep.axvline(0, color="k", lw=0.8)
    ax_dep.set_xlim(0, t_end / 60)
    ax_dep.set_ylim(0, dmax * 1.5)
    ax_dep.set_xlabel("Time (min)")
    ax_dep.set_ylabel("Max depth on asset (m)")
    ax_dep.set_title("Depth at protected assets" + (f" (objective J: {b['j']:.4f} → {a['j']:.4f})" if a else ""), pad=6)
    ax_dep.legend(loc="upper left", fontsize=9.5, ncol=len(series))
    times = frame_times(args, t_end)
    live = [ax_rain.fill_between([], [], lw=0)]

    def draw(i):
        t = times[i]
        db, da = lerp(t, ts, hb), lerp(t, ts, ha)
        depth_ims[0].set_data(np.ma.masked_less(db, 0.005))
        depth_ims[1].set_data(np.ma.masked_less(da, 0.005))
        diff = da - db
        depth_ims[2].set_data(np.ma.masked_where((np.abs(diff) < 0.003) | ((db < 0.005) & (da < 0.005)), diff))
        for qv, q, d in ((quivers[0], qb, db), (quivers[1], qa, da)):
            qq = lerp(t, ts, q)[::sub, ::sub]
            mag = np.hypot(qq[..., 0], qq[..., 1])
            # Arrow length ~ sqrt(discharge), capped at 16 m, so slow sheet flow stays visible next to channel flow.
            s = np.where((mag > 1e-3) & (d[stride // 2::stride * sub, stride // 2::stride * sub][:mag.shape[0], :mag.shape[1]] > 0.005), np.minimum(120 * np.sqrt(mag), 16) / np.maximum(mag, 1e-12), np.nan)
            qv.set_UVC(qq[..., 0] * s, qq[..., 1] * s)
        rain_cur.set_xdata([t / 60])
        dep_cur.set_xdata([t / 60])
        m = tr <= t
        live[0].remove()
        live[0] = ax_rain.fill_between(tr[m] / 60, rain[m], color="#1f4e9c", alpha=0.55, lw=0)
        for lb, la, sb, sa in lines:
            m2 = ts <= t
            tt = np.append(ts[m2], t) / 60
            lb.set_data(tt, np.append(sb[m2], np.interp(t, ts, sb)))
            la.set_data(tt, np.append(sa[m2], np.interp(t, ts, sa)))
        mm, ss = divmod(int(round(t)), 60)
        clock.set_text(f"t = {mm:02d}:{ss:02d}  rain {np.interp(t, hyeto[:, 0], hyeto[:, 1], right=0):4.0f} mm/h")
        fig.canvas.draw()
        return fig.canvas.buffer_rgba().tobytes()

    return draw, len(times), (W, H)


def _worker(args, start, stop, path):
    draw, _, size = build(args)
    w = videoio.open_writer(path, size, args.fps, "rgba", args.encoder)
    for i in range(start, stop):
        w.stdin.write(draw(i))
    videoio.close_writer(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="diverted-flood")
    ap.add_argument("--run", type=Path, help="itr optimize output (default runs/readme/<example>)")
    ap.add_argument("--out", type=Path, help="default docs/video/<example>.mp4")
    ap.add_argument("--seconds", type=float, default=16.0, help="duration of the simulated-time sweep")
    ap.add_argument("--hold", type=float, default=2.5, help="still frames at the end (s)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=videoio.default_jobs(), help="parallel render processes")
    ap.add_argument("--encoder", choices=["auto", "videotoolbox", "x264"], default="auto")
    args = ap.parse_args()
    out = args.out or ROOT / "docs" / "video" / f"{args.example}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(args.seconds * args.fps)) + int(round(args.hold * args.fps))
    videoio.render(n, args.jobs, _worker, args, out)


if __name__ == "__main__":
    main()
