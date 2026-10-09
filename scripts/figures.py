#!/usr/bin/env python3
"""Report figures for the README (docs/img/fig*.png).

Not part of the engine; needs numpy, matplotlib and tifffile:

    uv run --with numpy --with matplotlib --with tifffile scripts/figures.py

Inputs (produced by `itr` and scripts/lever_study.py):
    runs/readme/diverted-flood, runs/readme/synthetic-valley   `itr optimize` output directories
    docs/data/levers.json, docs/data/optimizers.json            scripts/lever_study.py
    docs/data/bench.json                                         `itr bench --json` over thread counts
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from matplotlib.colors import LightSource, LogNorm, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"
DATA = ROOT / "docs" / "data"
RUNS = ROOT / "runs" / "readme"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "figure.facecolor": "white",
})

METHOD_LABEL = {"random": "Random", "sobol": "Sobol (scrambled)", "cma_es": "CMA-ES", "lq_cma_es": "lq-CMA-ES"}
METHOD_STYLE = {"random": ("0.55", ":"), "sobol": ("0.35", "-."), "cma_es": ("0.0", "-"), "lq_cma_es": ("#1f4e9c", "--")}


def panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")


def read_tif(p):
    with tifffile.TiffFile(p) as t:
        page = t.pages[0]
        a = page.asarray().astype(float)
        sx, sy, _ = page.tags["ModelPixelScaleTag"].value
        tp = page.tags["ModelTiepointTag"].value
    a[a <= -9998] = np.nan
    x0, y0 = tp[3] - tp[0] * sx, tp[4] + tp[1] * sy
    ny, nx = a.shape
    # Extent relative to the grid origin (south-west corner at 0, 0), in metres.
    return a, (0, nx * sx, 0, ny * sy), (x0, y0 - ny * sy)


def polys(geojson, origin):
    out = []
    for f in json.loads(Path(geojson).read_text())["features"]:
        g = f["geometry"]
        rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
        for r in rings:
            r = np.asarray(r)
            out.append((r[:, 0] - origin[0], r[:, 1] - origin[1]))
    return out


def scalebar(ax, extent, length):
    x0 = extent[0] + 0.05 * (extent[1] - extent[0])
    y0 = extent[2] + 0.05 * (extent[3] - extent[2])
    ax.plot([x0, x0 + length], [y0, y0], color="k", lw=2, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + 0.02 * (extent[3] - extent[2]), f"{length:g} m", ha="center", va="bottom", fontsize=7)


def fig_maps(run, example, out):
    z, ext, origin = read_tif(run / "terrain_after.tif")
    zb, _, _ = read_tif(run / "terrain_before.tif")
    pb, _, _ = read_tif(run / "peak_depth_before.tif")
    pa, _, _ = read_tif(run / "peak_depth_after.tif")
    dd, _, _ = read_tif(run / "delta_peak_depth.tif")
    cell = (ext[1] - ext[0]) / z.shape[1]
    ls = LightSource(azdeg=315, altdeg=45)
    assets = polys(ROOT / "examples" / example / "protected.geojson", origin)
    works = polys(run / "earthworks.geojson", origin)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    norm = LogNorm(0.005, 2.0)
    for ax, terr, h, title, letter in [(axs[0], zb, pb, "No change", "a"), (axs[1], z, pa, "Optimized design", "b")]:
        hs = ls.hillshade(np.nan_to_num(terr, nan=np.nanmax(terr)), vert_exag=3, dx=cell, dy=cell)
        ax.imshow(hs, cmap="gray", extent=ext, vmin=-0.8, vmax=1.25)
        im = ax.imshow(np.ma.masked_less(h, 0.005), cmap="Blues", norm=norm, extent=ext, alpha=0.9, interpolation="nearest")
        ax.set_title(f"Peak depth, {title.lower()}")
        panel(ax, letter)
    cb = fig.colorbar(im, ax=axs[:2], orientation="horizontal", fraction=0.06, pad=0.02, aspect=40)
    cb.set_label("Peak water depth (m), log scale")
    hs = ls.hillshade(np.nan_to_num(z, nan=np.nanmax(z)), vert_exag=3, dx=cell, dy=cell)
    axs[2].imshow(hs, cmap="gray", extent=ext, vmin=-0.8, vmax=1.25)
    lim = 0.2
    im2 = axs[2].imshow(np.ma.masked_inside(dd, -0.005, 0.005), cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), extent=ext, interpolation="nearest")
    axs[2].set_title("Difference (b) − (a)")
    panel(axs[2], "c")
    cb2 = fig.colorbar(im2, ax=axs[2], orientation="horizontal", fraction=0.06, pad=0.02, aspect=20, extend="both")
    cb2.set_label("Δ peak depth (m)")
    for k, ax in enumerate(axs):
        for x, y in assets:
            ax.plot(x, y, color="k", lw=0.9)
        if k:
            for x, y in works:
                ax.plot(x, y, color="#b2182b" if k == 1 else "k", lw=0.9, ls="--")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("Easting (m)")
        if k == 0:
            ax.set_ylabel("Northing (m)")
        else:
            ax.set_yticklabels([])
        ax.set_aspect("equal")
        scalebar(ax, ext, 100)
    fig.savefig(out)
    plt.close(fig)


def best_so_far(trace, n):
    y = np.full(n, np.nan)
    best = np.nan
    for k, e in enumerate(trace[:n]):
        if e["feasible"] and (np.isnan(best) or e["j"] < best):
            best = e["j"]
        y[k] = best
    return y


def fig_convergence(opt, out):
    scen = list(opt["scenarios"])
    fig, axs = plt.subplots(1, len(scen), figsize=(7.2, 2.6), constrained_layout=True)
    for ax, s, letter in zip(axs, scen, "abcd"):
        base = None
        for m, runs in opt["scenarios"][s].items():
            n = min(len(r["trace"]) for r in runs)
            Y = np.array([best_so_far(r["trace"], n) for r in runs])
            base = runs[0]["j_baseline"]
            Y = np.where(np.isnan(Y), base, Y)
            x = np.arange(1, n + 1)
            c, ls = METHOD_STYLE[m]
            med = np.median(Y, 0)
            ax.plot(x, med, color=c, ls=ls, lw=1.1, label=METHOD_LABEL[m])
            ax.fill_between(x, np.percentile(Y, 25, 0), np.percentile(Y, 75, 0), color=c, alpha=0.12, lw=0)
        ax.axhline(base, color="k", lw=0.6, ls=(0, (1, 2)))
        ax.text(1.5, base, "no change", va="bottom", fontsize=7)
        ax.set_xlabel("Evaluation index")
        ax.set_ylabel("Best feasible objective $J$")
        ax.set_title(f"{s} ({len(runs)} seeds, median and IQR)")
        ax.set_xlim(1, None)
        panel(ax, letter)
    axs[0].legend(loc="upper right")
    fig.savefig(out)
    plt.close(fig)


def fig_levers(lev, out):
    scen = list(lev["scenarios"])
    fig, axs = plt.subplots(1, len(scen), figsize=(7.2, 2.5), constrained_layout=True)
    for ax, s, letter in zip(axs, scen, "abcd"):
        rows = lev["scenarios"][s]
        ref = rows[0]
        names = [r["variant"] for r in rows][::-1]
        cu = [r["cell_updates"] / ref["cell_updates"] for r in rows][::-1]
        wt = [r["search_wall_s"] / ref["search_wall_s"] for r in rows][::-1]
        y = np.arange(len(names))
        ax.barh(y + 0.19, cu, height=0.36, color="0.25", label="cell-updates")
        ax.barh(y - 0.19, wt, height=0.36, color="white", edgecolor="0.25", lw=0.6, label="search wall time")
        for k, r in enumerate(rows[::-1]):
            j = r["j_best"]
            ax.text(max(cu[k], wt[k]) * 1.04 + 0.02, y[k], f"$J$={j:.4f}" if j is not None else "infeasible", va="center", fontsize=7)
        ax.axvline(1, color="k", lw=0.6)
        ax.set_yticks(y, [n.replace("_", " ") for n in names])
        ax.set_xlabel("Relative to reference (= 1)")
        ax.set_title(s)
        ax.set_xlim(0, max(max(cu), max(wt)) * 1.45)
        ax.tick_params(axis="y", which="both", right=False, left=False)
        panel(ax, letter)
    axs[0].legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)


def fig_revalidation(runs, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    labels, before, after = [], [], []
    for name, run in runs:
        nominal = json.loads((run / "metrics.json").read_text())
        labels.append(f"{name}\nnominal")
        before.append(nominal["baseline"]["j"])
        after.append(nominal["candidate"]["j"])
        for c in json.loads((run / "revalidation.json").read_text())["cases"]:
            if "skipped" in c:
                continue
            labels.append(f"{name}\n{c['case'].replace('finer_grid_2x', 'grid ½Δx').replace('rain_x', 'rain ×')}")
            before.append(c["j_before"])
            after.append(c["j_after"])
    rel = [a / b for a, b in zip(after, before)]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, rel, color="0.3", height=0.6)
    ax.axvline(1, color="k", lw=0.6, ls="--")
    ax.set_yticks(y, [l.replace("\n", ": ") for l in labels], fontsize=7)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("$J_\\mathrm{design} / J_\\mathrm{no\\,change}$")
    ax.tick_params(axis="y", which="both", right=False, left=False)
    fig.savefig(out)
    plt.close(fig)


def fig_mass(run, out):
    d = np.genfromtxt(run / "mass_balance_before.csv", delimiter=",", names=True)
    t = d["t_s"] / 60
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), constrained_layout=True)
    ax = axs[0]
    ax.plot(t, d["rain_m3"], "k-", lw=1, label="rain (cumulative)")
    ax.plot(t, d["volume_m3"], "k--", lw=1, label="storage")
    ax.plot(t, d["infiltration_m3"], color="0.5", lw=1, label="infiltration")
    ax.plot(t, d["outflow_m3"], color="0.5", ls=":", lw=1.2, label="outflow")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Volume (m³)")
    ax.legend(loc="upper left")
    panel(ax, "a")
    ax = axs[1]
    gross = np.maximum(d["rain_m3"] + d["inflow_m3"], 1e-30)
    ax.semilogy(t, np.abs(d["residual_m3"]) / gross, "k.-", lw=0.8, ms=3)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("|residual| / gross input")
    ax.set_ylim(1e-10, 1e-3)
    ax.axhline(1e-4, color="0.5", lw=0.6, ls="--")
    ax.text(t[-1], 1.2e-4, "target $10^{-4}$", ha="right", fontsize=7)
    panel(ax, "b")
    fig.savefig(out)
    plt.close(fig)


def fig_bench(bench, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    pts = {}
    for c in bench:
        pts.setdefault((c["nx"], c["precision"]), {})[c["threads"]] = c["cell_updates_per_s"]
    for (n, p), d in sorted(pts.items()):
        th = sorted(d)
        ax.plot(th, [d[k] / 1e6 for k in th], marker="o" if p == "f32" else "s", ms=3.5, lw=0.9,
                color="k" if n == max(k[0] for k in pts) else "0.55", ls="-" if p == "f32" else "--",
                mfc="white" if p == "f64" else None, label=f"{n}², {p}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Threads (grid-parallel)")
    ax.set_ylabel("Cell-updates s$^{-1}$ ($\\times 10^6$)")
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(out)
    plt.close(fig)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    df, sv = RUNS / "diverted-flood", RUNS / "synthetic-valley"
    fig_maps(df, "diverted-flood", IMG / "fig1_maps.png")
    fig_convergence(json.loads((DATA / "optimizers.json").read_text()), IMG / "fig2_convergence.png")
    fig_levers(json.loads((DATA / "levers.json").read_text()), IMG / "fig3_levers.png")
    fig_revalidation([("diverted-flood", df), ("synthetic-valley", sv)], IMG / "fig4_revalidation.png")
    fig_mass(df, IMG / "fig5_mass_balance.png")
    bench = DATA / "bench.json"
    if bench.exists():
        fig_bench(json.loads(bench.read_text()), IMG / "fig6_throughput.png")


if __name__ == "__main__":
    main()
