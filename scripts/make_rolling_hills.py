#!/usr/bin/env python3
"""Write the `rolling-hills` example (examples/rolling-hills): a more varied synthetic
terrain than the bundled `itr synth` examples, used for the simulation videos.

Hills, two gullies that join above a village, a single channel draining south and
multi-scale roughness, on a 120 × 96 grid at 5 m. Deterministic (fixed seed).

    uv run --with numpy --with tifffile scripts/make_rolling_hills.py
"""

import json
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "rolling-hills"
X0, Y0 = 500_000.0, 4_200_000.0
NX, NY, DX = 120, 96, 5.0

# Polylines (local metres, south-west origin) carved as gullies.
GULLY_W = [(110, 480), (150, 400), (210, 330), (255, 260), (290, 205)]
GULLY_E = [(500, 480), (450, 410), (400, 330), (345, 255), (300, 205)]
# Below the confluence the stream is shallow and wide (an alluvial fan), so it spills
# over the ground the village stands on.
CHANNEL = [(290, 205), (285, 150), (300, 95), (330, 40), (345, 0)]
# Shallow bypass swale from beside the confluence to the relief route, behind a low lip.
BYPASS = [(335, 205), (390, 182), (450, 152), (500, 125)]
# A shallow relief route east of the village, separated from the east gully by a low saddle.
EAST_ROUTE = [(418, 296), (470, 220), (505, 130), (530, 60), (545, 0)]
HILLS = [(120, 380, 6.0, 60), (480, 395, 5.0, 70), (300, 330, 3.5, 40), (70, 160, 4.0, 55), (580, 200, 4.5, 50), (420, 110, 3.0, 40)]

HOUSES = [  # (name, x0, y0, x1, y1), on the fan below the confluence
    ("house_a", 285, 160, 303, 172),
    ("house_b", 300, 128, 318, 140),
    ("house_c", 275, 100, 293, 112),
    ("house_d", 295, 58, 313, 70),
]
GUARD = ("west_farm", 150, 20, 240, 80)
EDITABLE = ("upper_fields", 200, 145, 470, 340)  # house footprints are cut out


EARTHWORKS = """# Earthwork primitives (spec §6.1). Bounds are [min, max].
[[primitive]]
kind = "berm"
height_m = [0.0, 1.0]
width_m = [4.0, 10.0]
length_m = [20.0, 160.0]
angle_deg = [0.0, 180.0]

[[primitive]]
kind = "berm"
height_m = [0.0, 1.0]
width_m = [4.0, 10.0]
length_m = [20.0, 160.0]
angle_deg = [0.0, 180.0]

[[primitive]]
kind = "swale"
height_m = [0.0, 0.6]
width_m = [4.0, 10.0]
length_m = [20.0, 160.0]
angle_deg = [0.0, 180.0]
"""


def polyline_dist(px, py, pts):
    d = np.full(px.shape, np.inf)
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        vx, vy = bx - ax, by - ay
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        d = np.minimum(d, np.hypot(px - ax - t * vx, py - ay - t * vy))
    return d


def smooth_noise(rng, scale_cells):
    f = rng.standard_normal((NY, NX))
    ky, kx = np.meshgrid(np.fft.fftfreq(NY), np.fft.fftfreq(NX), indexing="ij")
    g = np.exp(-0.5 * (kx * kx + ky * ky) * (2 * np.pi * scale_cells) ** 2)
    n = np.real(np.fft.ifft2(np.fft.fft2(f) * g))
    return n / n.std()


def terrain():
    x = (np.arange(NX) + 0.5) * DX
    y = (NY - np.arange(NY) - 0.5) * DX  # rows north-first
    px, py = np.meshgrid(x, y)
    z = 60.0 + 0.03 * py + 0.002 * (px - 330) ** 2 / 30
    for hx, hy, a, s in HILLS:
        z += a * np.exp(-((px - hx) ** 2 + (py - hy) ** 2) / (2 * s * s))
    # Gullies fade out as they reach the fan, so their flow spreads instead of pooling.
    fade = np.clip((py - 185) / 110, 0.15, 1)
    for line, depth, w in ((GULLY_W, 2.2, 14), (GULLY_E, 2.0, 13)):
        z -= fade * depth * np.exp(-(polyline_dist(px, py, line) / w) ** 2)
    for line, depth, w in ((CHANNEL, 0.35, 26), (EAST_ROUTE, 1.4, 16), (BYPASS, 0.8, 12)):
        z -= depth * np.exp(-(polyline_dist(px, py, line) / w) ** 2)
    # Fan: smooth ground either side of the lower channel.
    fan = np.exp(-(polyline_dist(px, py, CHANNEL) / 70) ** 2) / (1 + np.exp((py - 200) / 12))
    rng = np.random.default_rng(20261009)
    z += (1 - 0.85 * fan) * (0.9 * smooth_noise(rng, 9) + 0.35 * smooth_noise(rng, 4)) + 0.08 * smooth_noise(rng, 1.5)
    return z.astype(np.float32)


def rect(x0, y0, x1, y1):
    return [[[X0 + x0, Y0 + y0], [X0 + x1, Y0 + y0], [X0 + x1, Y0 + y1], [X0 + x0, Y0 + y1], [X0 + x0, Y0 + y0]]]


def geojson(path, feats, holes=()):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": rect(*box) + [rect(*h)[0] for h in holes]}}
        for box, props in feats]}, indent=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = terrain()
    tifffile.imwrite(OUT / "dem.tif", z, compression="deflate", extratags=[
        (33550, "d", 3, (DX, DX, 0.0)),
        (33922, "d", 6, (0.0, 0.0, 0.0, X0, Y0 + NY * DX, 0.0)),
        (34735, "H", 16, (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32643)),
        (42113, "s", 0, "-9999"),
    ])
    geojson(OUT / "protected.geojson", [(h[1:], {"name": h[0], "threshold_m": 0.05}) for h in HOUSES])
    geojson(OUT / "guard.geojson", [(GUARD[1:], {"name": GUARD[0]})])
    ex0, ey0, ex1, ey1 = EDITABLE[1:]
    inside = [h[1:] for h in HOUSES if h[1] < ex1 and h[3] > ex0 and h[2] < ey1 and h[4] > ey0]
    geojson(OUT / "editable.geojson", [(EDITABLE[1:], {"name": EDITABLE[0]})], holes=inside)
    (OUT / "earthworks.toml").write_text(EARTHWORKS)
    (OUT / "scenario.toml").write_text(f'''schema_version = "0.2"
scenario_id = "rolling-hills-001"

[terrain]
path = "dem.tif"
crs = "EPSG:32643"
vertical_datum = "synthetic-local"

[hydrology]
duration_s = 2700
sync_interval_s = 60
rainfall_hyetograph = [[0, 0], [300, 85], [1200, 45], [1800, 0]]
manning_n = 0.045

[hydrology.infiltration]
model = "constant_capacity"
capacity_mm_h = 8.0

[boundaries]
segments = [{{ kind = "transmissive", side = "south" }}]

[design]
editable_mask = "editable.geojson"
primitives = "earthworks.toml"
max_abs_elevation_change_m = 1.0
max_earthwork_volume_m3 = 3000
placement = "corridor"

[objectives]
protected_areas = "protected.geojson"
downstream_guard_areas = "guard.geojson"
depth_threshold_m = 0.05
guard_tolerance_m = 0.01

[optimizer]
seed = 7
max_simulations = 300
fidelity_levels = [1]
''')
    print(OUT, f"z {z.min():.1f}..{z.max():.1f} m")


if __name__ == "__main__":
    main()
