#!/usr/bin/env python3
"""Lever study (spec §7.11.7, Phase D exit report) and optimizer comparison.

Runs `itr optimize` on the example scenarios with one lever changed at a time and
records wall time, cell-updates, simulations and the best feasible J. The optimizer
comparison repeats each method over several seeds at an equal simulation budget.

Standard library only. Usage:

    cargo build --release
    python3 scripts/lever_study.py [--seeds 5] [--out docs/data]

Writes <out>/levers.json, <out>/optimizers.json and per-run search traces under
runs/study/. Wall times depend on the host and its load; the JSON records both.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ITR = ROOT / "target" / "release" / "itr"
RUNS = ROOT / "runs" / "study"


def set_key(text: str, section: str, key: str, value: str) -> str:
    """Set `key = value` in `[section]` of a flat TOML file (appends the section if absent)."""
    lines = text.splitlines()
    head = f"[{section}]"
    if head not in (l.strip() for l in lines):
        return text.rstrip() + f"\n\n{head}\n{key} = {value}\n"
    out, inside, done = [], False, False
    for l in lines:
        s = l.strip()
        if s.startswith("["):
            if inside and not done:
                out.append(f"{key} = {value}")
                done = True
            inside = s == head
        elif inside and re.match(rf"{re.escape(key)}\s*=", s):
            if not done:
                out.append(f"{key} = {value}")
                done = True
            continue
        out.append(l)
    if inside and not done:
        out.append(f"{key} = {value}")
    return "\n".join(out) + "\n"


def variant(example: str, name: str, edits: list) -> Path:
    src = ROOT / "examples" / example
    dst = RUNS / "scenarios" / f"{example}--{name}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    p = dst / "scenario.toml"
    t = p.read_text()
    for sec, k, v in edits:
        t = set_key(t, sec, k, v)
    p.write_text(t)
    return p


def optimize(scen: Path, out: Path, extra: list) -> dict:
    if out.exists():
        shutil.rmtree(out)
    cmd = [str(ITR), "optimize", "--scenario", str(scen), "--out", str(out), "--no-revalidate", *extra]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode not in (0, 2):
        sys.exit(f"{' '.join(cmd)} failed:\n{r.stderr}")
    m = json.loads((out / "metrics.json").read_text())
    trace = [json.loads(l) for l in (out / "search.jsonl").read_text().splitlines() if l.strip()]
    status = {}
    for e in trace:
        s = e["record"]["status"]
        status[s] = status.get(s, 0) + 1
    b = m["budget"]
    return {
        "outcome": m["outcome"],
        "j_baseline": m["baseline"]["j"],
        "j_best": m.get("candidate", {}).get("j"),
        "simulations": b["simulations"],
        "cell_updates": b["cell_updates"],
        "search_wall_s": b["search_wall_s"],
        "utilization": b["utilization"],
        "total_wall_s": time.time() - t0,
        "status_counts": status,
        "trace": [
            {"i": e["eval_index"], "level": e["level"], "j": e["record"]["j"],
             "feasible": e["record"]["fitness_feasible"] and e["record"]["status"] == "ok"}
            for e in trace
        ],
    }


LEVERS = {
    # (example, variant name, scenario edits, extra CLI args, description)
    "synthetic-valley": [
        ("reference", [], [], "example as shipped: CMA-ES, levels [2, 1], adaptive Δt"),
        ("single_level", [("optimizer", "fidelity_levels", "[1]"), ("optimizer", "level_budget_share", "[1.0]")], [],
         "multi-fidelity off: all simulations at level 1"),
        ("local_inertial", [("solver", "screening_physics", '"local_inertial"')], [], "local-inertial physics on level 2"),
        ("locked_dt", [("solver", "dt_mode", '"baseline_locked"')], [], "baseline-locked Δt"),
        ("locked_dt_replay", [("solver", "dt_mode", '"baseline_locked"'), ("solver", "subdomain_replay", '"auto"')], [],
         "baseline-locked Δt + subdomain replay"),
        ("threads_1", [], ["--threads", "1"], "one worker (candidate parallelism off)"),
        ("f64", [], ["--precision", "f64"], "f64 oracle precision"),
    ],
    "diverted-flood": [
        ("reference", [], [], "example as shipped: CMA-ES, corridor placement, level 1"),
        ("free_placement", [("design", "placement", '"free"')], [], "corridor design space off"),
        ("locked_dt", [("solver", "dt_mode", '"baseline_locked"')], [], "baseline-locked Δt"),
        ("locked_dt_replay", [("solver", "dt_mode", '"baseline_locked"'), ("solver", "subdomain_replay", '"auto"')], [],
         "baseline-locked Δt + subdomain replay"),
        ("threads_1", [], ["--threads", "1"], "one worker (candidate parallelism off)"),
        ("f64", [], ["--precision", "f64"], "f64 oracle precision"),
    ],
}

METHODS = ["random", "sobol", "cma_es", "lq_cma_es"]


def host() -> dict:
    cpu = platform.processor()
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or cpu
    except OSError:
        pass
    return {"cpu": cpu, "os": platform.platform(), "logical_cpus": os.cpu_count(),
            "load_average_at_start": os.getloadavg() if hasattr(os, "getloadavg") else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "docs" / "data"))
    ap.add_argument("--skip-levers", action="store_true")
    ap.add_argument("--skip-optimizers", action="store_true")
    a = ap.parse_args()
    if not ITR.exists():
        sys.exit("build first: cargo build --release")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    if not a.skip_levers:
        res = {"host": host(), "scenarios": {}}
        for ex, rows in LEVERS.items():
            res["scenarios"][ex] = []
            for name, edits, extra, desc in rows:
                print(f"lever {ex} {name}", file=sys.stderr)
                r = optimize(variant(ex, name, edits), RUNS / f"lever-{ex}-{name}", extra)
                r.pop("trace")
                res["scenarios"][ex].append({"variant": name, "description": desc, **r})
        (out / "levers.json").write_text(json.dumps(res, indent=1))

    if not a.skip_optimizers:
        res = {"host": host(), "seeds": a.seeds, "scenarios": {}}
        for ex in ["synthetic-valley", "diverted-flood"]:
            res["scenarios"][ex] = {}
            for m in METHODS:
                runs = []
                for s in range(a.seeds):
                    print(f"optimizer {ex} {m} seed {s}", file=sys.stderr)
                    r = optimize(ROOT / "examples" / ex / "scenario.toml", RUNS / f"opt-{ex}-{m}-{s}",
                                 ["--method", m, "--seed", str(1000 + s)])
                    runs.append({"seed": 1000 + s, **r})
                res["scenarios"][ex][m] = runs
        (out / "optimizers.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
