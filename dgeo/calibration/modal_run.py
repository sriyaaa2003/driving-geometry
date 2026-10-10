"""Run the real-data calibration study on Modal GPUs, one container per Argoverse 2 log.

    modal run dgeo/calibration/modal_run.py --n-logs 16 --out results/calibration_av2

Each container downloads its log from the open Argoverse 2 bucket, builds the camera pairs and runs the
solver on a GPU. The trial table comes back as <out>/trials.csv; `python -m dgeo.calibration.report`
turns it into the tables and figure.
"""
from __future__ import annotations

from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy<2", "scipy", "pandas", "pyarrow<17", "opencv-python-headless<4.11")
    .add_local_python_source("dgeo")
)
app = modal.App("dgeo-calibration-av2", image=image)

CAM_A = "ring_front_center"
CAMS_B = ["stereo_front_left", "stereo_front_right"]
SEG_FRAMES, N_TRIALS = 10, 3


@app.function(gpu="T4", timeout=3 * 3600, retries=1)
def run_log(log_id: str, split: str = "val") -> str:
    import pandas as pd
    import torch

    from dgeo.calibration import av2, study

    log = av2.load_log(log_id, Path("/tmp/av2"), [CAM_A, *CAMS_B], split=split, every=3, max_frames=60)
    rows = []
    for s in range(0, len(log.frames) - SEG_FRAMES + 1, SEG_FRAMES):
        frames = log.frames[s : s + SEG_FRAMES]
        for cam_b in CAMS_B:
            for r in study.run_segment(log, frames, CAM_A, cam_b, "cuda", 4, N_TRIALS, seed=s):
                rows.append({"log_id": log_id, "segment": s // SEG_FRAMES, **r})
    torch.cuda.empty_cache()
    return pd.DataFrame(rows).to_csv(index=False)


@app.local_entrypoint()
def main(n_logs: int = 16, out: str = "results/calibration_av2", split: str = "val") -> None:
    import io

    import pandas as pd

    from dgeo.calibration import av2

    logs = av2.list_logs(split)
    step = max(1, len(logs) // n_logs)
    chosen = logs[::step][:n_logs]
    frames = []
    for log_id, csv in zip(chosen, run_log.map(chosen, kwargs={"split": split}, return_exceptions=True), strict=True):
        if isinstance(csv, Exception):
            print("FAILED", log_id, type(csv).__name__, csv)
            continue
        d = pd.read_csv(io.StringIO(csv))
        print(log_id, len(d), "trials")
        frames.append(d)
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(Path(out) / "trials.csv", index=False)
    print("wrote", Path(out) / "trials.csv")
