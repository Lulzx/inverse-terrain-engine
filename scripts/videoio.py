"""Video output shared by the simulation-video scripts.

Frames are rendered in parallel worker processes, each writing a contiguous chunk of
frames to its own H.264 segment; the segments are then joined without re-encoding.
Encoding uses the VideoToolbox hardware encoder when ffmpeg has it (Apple GPUs/media
engine), otherwise libx264.
"""

import functools
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


@functools.cache
def has_videotoolbox():
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    return "h264_videotoolbox" in out


def encoder_args(encoder):
    if encoder == "auto":
        encoder = "videotoolbox" if has_videotoolbox() else "x264"
    if encoder == "videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", "14M", "-maxrate", "20M", "-bufsize", "28M", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]


def open_writer(path, size, fps, pix_fmt, encoder):
    """ffmpeg process reading raw frames of `pix_fmt` on stdin."""
    w, h = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", *encoder_args(encoder), "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def close_writer(proc):
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg exited with {proc.returncode}")


def default_jobs():
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def render(n_frames, jobs, worker, worker_args, out):
    """Render frames [0, n_frames) with `worker(worker_args, start, stop, path)` into `out`.

    The worker must produce identical frames for a given index regardless of the chunk
    it runs in, so the joined video does not depend on `jobs`.
    """
    t0 = time.perf_counter()
    jobs = max(1, min(jobs, n_frames))
    if jobs == 1:
        worker(worker_args, 0, n_frames, out)
    else:
        tmp = Path(tempfile.mkdtemp(prefix="itr-video-"))
        try:
            bounds = [round(k * n_frames / jobs) for k in range(jobs + 1)]
            segs = [tmp / f"seg{k:03d}.mp4" for k in range(jobs)]
            with mp.get_context("spawn").Pool(jobs) as pool:
                pool.starmap(worker, [(worker_args, bounds[k], bounds[k + 1], segs[k]) for k in range(jobs)])
            (tmp / "list.txt").write_text("".join(f"file '{s}'\n" for s in segs))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                            "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    dt = time.perf_counter() - t0
    print(f"{out}: {n_frames} frames in {dt:.1f} s ({n_frames / dt:.0f} fps, {jobs} jobs)")
