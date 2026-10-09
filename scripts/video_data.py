#!/usr/bin/env python3
"""Prepare the data the fframes renderer (video/) draws: re-run the no-change and the
optimized terrain with a short sync interval, then pack terrain and per-frame water into
textures.

Every video frame shows a recorded solver state; nothing is interpolated in time.
`--sync` sets the recording interval (default: the storm duration divided over the video
length, rounded to whole seconds). The recorded peaks are checked against the optimizer's
result before anything is written.

Outputs (runs/video/<example>/), raw RGBA8 with A = 255, rows north-first:
    terrain_{before,after}.rgba   RG: elevation (16 bit), B: material code      (UP*ny x UP*nx)
    depth_{before,after}.rgba     per frame, RG: water depth (16 bit over depth_span), B: 255 where wet
    vel_{before,after}.rgba       per frame, RG: velocity u, v (8 bit, ±VEL_MAX m/s), B: speed
    meta.json                     geometry, ranges, times, rain, buildings, series, result

    uv run --with numpy --with scipy --with tifffile --with lz4 scripts/video_data.py --example rolling-hills
"""

import argparse
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

import lz4.block
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent.parent
ITR = ROOT / "target" / "release" / "itr"
UP = 4  # terrain texture refinement over the simulation grid
VEL_MAX = 2.0  # m/s, velocity texture range
WET = 0.002  # m; shallower cells are drawn dry
HOUSE_WALL, HOUSE_ROOF = 3.0, 2.2  # m, display only


def read_tif(path):
    with tifffile.TiffFile(path) as t:
        page = t.pages[0]
        a = page.asarray().astype(np.float64)
        sx, sy, _ = page.tags["ModelPixelScaleTag"].value
        tp = page.tags["ModelTiepointTag"].value
    a[a <= -9998] = np.nan
    ny, nx = a.shape
    return a, (tp[3], tp[4] - ny * sy), sx  # rows north-first, south-west origin, cell size


def rings(path, origin):
    out = []
    for f in json.loads(Path(path).read_text())["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for p in polys:
            r = np.asarray(p[0])
            out.append({"props": f.get("properties", {}), "xy": (r - origin).round(3).tolist()})
    return out


def simulate(scenario, dem, sync, out):
    """`itr simulate` on `dem` with all paths absolute and the given sync interval."""
    sc = tomllib.loads(scenario.read_text())
    base = scenario.parent
    lines = []
    for line in scenario.read_text().splitlines():
        key = line.split("=")[0].strip()
        if key == "path" and line.strip().startswith("path"):
            line = f'path = "{dem}"'
        elif key == "sync_interval_s":
            line = f"sync_interval_s = {sync}"
        elif key in ("editable_mask", "primitives", "protected_areas", "downstream_guard_areas"):
            line = f'{key} = "{(base / sc["design" if key in ("editable_mask", "primitives") else "objectives"][key]).resolve()}"'
        lines.append(line)
    tmp = out.with_suffix(".toml")
    tmp.write_text("\n".join(lines) + "\n")
    if out.exists():
        shutil.rmtree(out)
    subprocess.run([str(ITR), "simulate", "--scenario", str(tmp), "--out", str(out)], check=True, capture_output=True)
    idx = json.loads((out / "viewer_index.json").read_text())
    meta = idx["baseline"]
    ny, nx = idx["ny"], idx["nx"]
    t = np.array([f["t"] for f in meta["frames"]])
    h = np.stack([np.frombuffer(lz4.block.decompress((out / "frames" / f["file"]).read_bytes()), "<u2").reshape(ny, nx) / 1000.0
                  for f in meta["frames"]])
    v = meta["velocity"]
    q = np.frombuffer(lz4.block.decompress((out / "frames" / v["file"]).read_bytes()), "<f4").reshape(len(t), v["ny"], v["nx"], 2)
    return t, h, q, v["stride"], idx["assets"]


def pack16(a, lo, span):
    u = np.clip(np.round((a - lo) / span * 65535), 0, 65535).astype(np.uint32)
    return (u >> 8).astype(np.uint8), (u & 255).astype(np.uint8)


def rgba(r, g, b):
    r = np.asarray(r)
    return np.dstack([r, g, b, np.full(r.shape, 255)]).astype(np.uint8)


def refine(a, up):
    """Cubic refinement of a cell-centred grid: output texel centres map to input cell coordinates."""
    ny, nx = a.shape
    yy = (np.arange(ny * up) + 0.5) / up - 0.5
    xx = (np.arange(nx * up) + 0.5) / up - 0.5
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return map_coordinates(a, [Y, X], order=3, mode="nearest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="rolling-hills")
    ap.add_argument("--run", type=Path, help="itr optimize output (default runs/readme/<example>)")
    ap.add_argument("--out", type=Path, help="default runs/video/<example>")
    ap.add_argument("--sync", type=float, help="recording interval in simulated seconds")
    ap.add_argument("--video-seconds", type=float, default=20.0, help="target length of the storm sweep")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    run = args.run or ROOT / "runs" / "readme" / args.example
    out = args.out or ROOT / "runs" / "video" / args.example
    scen_path = ROOT / "examples" / args.example / "scenario.toml"
    scen = tomllib.loads(scen_path.read_text())
    duration = scen["hydrology"]["duration_s"]
    sync = args.sync or max(1.0, round(duration / (args.video_seconds * args.fps)))
    out.mkdir(parents=True, exist_ok=True)
    for p in [*out.glob("*.png"), *out.glob("*.rgba")]:  # stale outputs
        p.unlink()

    zb, origin, cell = read_tif(run / "terrain_before.tif")
    za, _, _ = read_tif(run / "terrain_after.tif")
    ny, nx = zb.shape
    fill = np.nanmin(zb)
    zb, za = np.nan_to_num(zb, nan=fill), np.nan_to_num(za, nan=fill)
    metrics = json.loads((run / "metrics.json").read_text())

    sims = {}
    for tag, dem in (("before", run / "terrain_before.tif"), ("after", run / "terrain_after.tif")):
        sims[tag] = simulate(scen_path, dem.resolve(), sync, out / f"sim_{tag}")
        print(f"{tag}: {len(sims[tag][0])} frames every {sync:g} s")

    # The re-run must reproduce the optimizer's peaks (sync only changes where steps are clipped).
    assets = sims["before"][4]
    for tag, key in (("before", "baseline"), ("after", "candidate")):
        h = sims[tag][1].reshape(len(sims[tag][0]), -1)
        for a, ref in zip(assets, metrics[key]["assets"]):
            got = h[:, a["cells"]].max()
            tol = max(0.003, 0.03 * ref["peak_depth_m"])  # 1 mm frame quantization + sync sampling
            status = "ok" if abs(got - ref["peak_depth_m"]) <= tol else "MISMATCH"
            print(f"  {tag:6s} {a['name']:10s} peak {got:.4f} m, optimizer {ref['peak_depth_m']:.4f} m  {status}")
            if status != "ok":
                raise SystemExit("re-simulation does not reproduce the optimizer result")

    # Elevation range shared by terrain and water textures.
    houses = rings(ROOT / "examples" / args.example / "protected.geojson", origin)
    hmax = max(sims["before"][1].max(), sims["after"][1].max())
    z_lo = float(min(zb.min(), za.min()) - 0.5)
    z_span = float(max(zb.max(), za.max()) + max(hmax, HOUSE_WALL + HOUSE_ROOF) + 0.5 - z_lo)

    # Terrain textures: cubic-refined elevation with buildings; material code in B
    # (0 natural, 1..120 fill depth in cm, 121..200 cut depth, 250 + k building k).
    fy, fx = ny * UP, nx * UP
    Yc = (np.arange(fy) + 0.5) * cell / UP  # metres from the north edge
    Xc = (np.arange(fx) + 0.5) * cell / UP
    YY, XX = np.meshgrid(ny * cell - Yc, Xc, indexing="ij")  # northing from the south edge
    for tag, z in (("before", zb), ("after", za)):
        zf = refine(z, UP)
        dz = refine(z - zb, UP)
        mat = np.where(dz > 0.02, np.clip(np.round(dz * 100), 1, 120), np.where(dz < -0.02, np.clip(121 + np.round(-dz * 100), 121, 200), 0))
        for k, hs in enumerate(houses):
            xy = np.asarray(hs["xy"])
            x0, x1, y0, y1 = xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()
            inside = (XX >= x0) & (XX <= x1) & (YY >= y0) & (YY <= y1)
            ground = zf[inside].max()
            if (x1 - x0) >= (y1 - y0):
                across = 1 - np.abs(YY - (y0 + y1) / 2) / ((y1 - y0) / 2)
            else:
                across = 1 - np.abs(XX - (x0 + x1) / 2) / ((x1 - x0) / 2)
            roof = ground + HOUSE_WALL + HOUSE_ROOF * np.clip(across, 0, 1)
            zf = np.where(inside, roof, zf)
            mat = np.where(inside, 250 + k, mat)
        r, g = pack16(zf, z_lo, z_span)
        (out / f"terrain_{tag}.rgba").write_bytes(rgba(r, g, mat).tobytes())

    # Water textures per frame: depth (drawn on top of the refined terrain, so shorelines
    # follow the interpolated depth field) and velocity from the strided discharge field.
    depth_span = float(max(hmax * 1.02, 0.05))
    series = {}
    times = sims["before"][0]
    for tag, z in (("before", zb), ("after", za)):
        t, h, q, stride, _ = sims[tag]
        assert np.array_equal(t, times)
        vny, vnx = q.shape[1:3]
        rr = np.clip(((np.arange(ny) - stride // 2) / stride).round().astype(int), 0, vny - 1)
        cc = np.clip(((np.arange(nx) - stride // 2) / stride).round().astype(int), 0, vnx - 1)
        etas, vels = [], []
        for i in range(len(t)):
            r, g = pack16(np.where(h[i] > WET, h[i], 0.0), 0.0, depth_span)
            etas.append(rgba(r, g, np.where(h[i] > WET, 255, 0)))
            qq = q[i][rr][:, cc]
            hh = np.maximum(h[i], 0.01)
            u, v = qq[..., 0] / hh, qq[..., 1] / hh  # north-positive v
            u, v = np.where(h[i] > WET, u, 0), np.where(h[i] > WET, v, 0)
            enc = lambda a: np.clip(np.round(128 + a / VEL_MAX * 127), 1, 255)  # noqa: E731
            vels.append(rgba(enc(u), enc(v), np.clip(np.hypot(u, v) / VEL_MAX * 255, 0, 255)))
        (out / f"depth_{tag}.rgba").write_bytes(np.stack(etas).tobytes())
        (out / f"vel_{tag}.rgba").write_bytes(np.stack(vels).tobytes())
        hf = h.reshape(len(t), -1)
        series[tag] = [hf[:, a["cells"]].max(1).round(4).tolist() for a in assets]

    hy = np.array(scen["hydrology"]["rainfall_hyetograph"], float)
    meta = {
        "example": args.example,
        "grid": {"nx": nx, "ny": ny, "cell": cell, "up": UP, "width": nx * cell, "height": ny * cell},
        "z": {"lo": z_lo, "span": z_span, "min": float(zb.min()), "max": float(zb.max())},
        "depth_span": depth_span,
        "vel_max": VEL_MAX,
        "sync_s": sync,
        "times": times.tolist(),
        "rain": [float(np.interp(tt, hy[:, 0], hy[:, 1], right=0.0)) for tt in times],
        "rain_max": float(hy[:, 1].max()),
        "houses": [{"name": a["name"], "threshold": a["threshold_m"], "xy": hs["xy"]} for a, hs in zip(assets, houses)],
        "guards": rings(ROOT / "examples" / args.example / "guard.geojson", origin),
        "earthworks": rings(run / "earthworks.geojson", origin),
        "series": series,
        "result": {
            "j_before": metrics["baseline"]["j"], "j_after": metrics["candidate"]["j"],
            "guard_worsening_m": metrics["candidate"].get("guard_max_worsening_m", 0.0),
            "guard_tolerance_m": scen["objectives"]["guard_tolerance_m"],
            "fill_m3": metrics["candidate"].get("fill_m3", 0.0), "cut_m3": metrics["candidate"].get("cut_m3", 0.0),
        },
        "manning_n": scen["hydrology"]["manning_n"],
    }
    (out / "meta.json").write_text(json.dumps(meta))
    print(out, f"{len(times)} frames, z {z_lo:.1f} + {z_span:.1f} m")


if __name__ == "__main__":
    main()
