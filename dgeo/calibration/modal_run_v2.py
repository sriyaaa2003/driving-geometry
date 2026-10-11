"""Calibration study v2 on Modal: ground-truth consistency check, bias test, perturbation sweep, depth-range
test, curvature test and a PnP baseline, on Argoverse 2 drives.

    modal run dgeo/calibration/modal_run_v2.py::survey --out results/calibration_v2        # which logs / how consistent
    modal run dgeo/calibration/modal_run_v2.py::main --n-logs 60 --out results/calibration_v2
"""
from __future__ import annotations

import io
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.1", "numpy<2", "scipy", "pandas", "pyarrow<17", "opencv-python-headless<4.11")
    .add_local_python_source("dgeo")
)
app = modal.App("dgeo-calibration-v2", image=image)
results = modal.Volume.from_name("dgeo-results", create_if_missing=True)

CAM_A = "ring_front_center"
CAMS_B = ["stereo_front_left", "stereo_front_right"]
SEG_FRAMES = 10


@app.function(timeout=3600)
def survey_chunk(log_ids: list[str], split: str = "val") -> str:
    """Per log: camera B extrinsics relative to camera A, and how much the vehicle turns."""
    import numpy as np
    import pandas as pd
    from scipy.spatial.transform import Rotation

    from dgeo.calibration import av2

    rows = []
    for lid in log_ids:
        try:
            root = f"{av2.PREFIX}/{split}/{lid}"
            d = Path("/tmp/survey") / lid
            ext = pd.read_feather(av2._get(f"{av2.BASE}{root}/calibration/egovehicle_SE3_sensor.feather", d / "e.feather"))
            poses = pd.read_feather(av2._get(f"{av2.BASE}{root}/city_SE3_egovehicle.feather", d / "p.feather"))

            def T(cam):
                r = ext[ext.sensor_name == cam].iloc[0]
                M = np.eye(4)
                M[:3, :3] = Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()
                M[:3, 3] = [r.tx_m, r.ty_m, r.tz_m]
                return M

            yaw = np.unwrap(Rotation.from_quat(poses[["qx", "qy", "qz", "qw"]].to_numpy()).as_euler("zyx")[:, 0])
            row = dict(log_id=lid, total_abs_heading_deg=float(np.abs(np.diff(yaw)).sum() * 180 / np.pi),
                       net_heading_deg=float(abs(yaw[-1] - yaw[0]) * 180 / np.pi))
            for cam in CAMS_B:
                rel = np.linalg.inv(T(CAM_A)) @ T(cam)
                rv = Rotation.from_matrix(rel[:3, :3]).as_rotvec()
                row.update({f"{cam}_rot{i}": float(np.degrees(rv[i])) for i in range(3)})
                row.update({f"{cam}_t{i}": float(rel[i, 3] * 100) for i in range(3)})
            rows.append(row)
        except Exception as exc:  # keep the survey going
            rows.append(dict(log_id=lid, error=str(exc)[:100]))
    return pd.DataFrame(rows).to_csv(index=False)


@app.local_entrypoint()
def survey(out: str = "results/calibration_v2", split: str = "val", chunk: int = 25) -> None:
    import pandas as pd

    from dgeo.calibration import av2

    logs = av2.list_logs(split)
    parts = [logs[i : i + chunk] for i in range(0, len(logs), chunk)]
    frames = [pd.read_csv(io.StringIO(c)) for c in survey_chunk.map(parts, kwargs={"split": split})]
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(Path(out) / "survey.csv", index=False)
    print("wrote", Path(out) / "survey.csv", sum(len(f) for f in frames), "logs")


@app.function(gpu="T4", timeout=3 * 3600, retries=1, volumes={"/results": results})
def run_log(log_id: str, split: str = "val") -> str:
    import pandas as pd
    import torch

    from dgeo.calibration import av2, baseline, study

    log = av2.load_log(log_id, Path("/tmp/av2"), [CAM_A, *CAMS_B], split=split, every=3, max_frames=60)
    rows = []
    for s in range(0, len(log.frames) - SEG_FRAMES + 1, SEG_FRAMES):
        frames = log.frames[s : s + SEG_FRAMES]
        for cam_b in CAMS_B:
            def go(**kw):
                return study.run_segment(log, frames, CAM_A, cam_b, "cuda", 4, seed=s, **kw)

            out = []
            out += go(n_trials=3, tag="main")
            out += go(n_trials=1, start_at_truth=True, tag="truth_start")
            for f in (0.5, 2.0, 4.0):
                out += go(n_trials=2, rot_deg=0.5 * f, trans_cm=5.0 * f, tag=f"perturb_x{f:g}")
            for name, (z0, z1) in {"near": (4, 15), "mid": (15, 30), "far": (30, 50)}.items():
                out += go(n_trials=2, zmin=z0, zmax=z1, tag=f"depth_{name}")
            out.append(baseline.pnp_baseline(log, frames, CAM_A, cam_b, 4))
            for r in out:
                rows.append({"log_id": log_id, "segment": s // SEG_FRAMES, **r})
    torch.cuda.empty_cache()
    csv = pd.DataFrame(rows).to_csv(index=False)
    Path("/results/calibration_v2").mkdir(parents=True, exist_ok=True)
    Path(f"/results/calibration_v2/{log_id}.csv").write_text(csv)
    results.commit()
    return csv


@app.local_entrypoint()
def main(n_logs: int = 60, out: str = "results/calibration_v2", survey_csv: str = "results/calibration_v2/survey.csv",
         split: str = "val") -> None:
    import pandas as pd

    sv = pd.read_csv(survey_csv).dropna(subset=["total_abs_heading_deg"])
    curved = sv.sort_values("total_abs_heading_deg", ascending=False).log_id.tolist()[: n_logs // 2]
    rest = [x for x in sv.log_id if x not in set(curved)]
    step = max(1, len(rest) // (n_logs - len(curved)))
    chosen = curved + rest[::step][: n_logs - len(curved)]
    frames = []
    for lid, csv in zip(chosen, run_log.map(chosen, kwargs={"split": split}, return_exceptions=True), strict=True):
        if isinstance(csv, Exception):
            print("FAILED", lid, type(csv).__name__, str(csv)[:150])
            continue
        d = pd.read_csv(io.StringIO(csv))
        print(lid, len(d), "rows", flush=True)
        frames.append(d)
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(Path(out) / "trials.csv", index=False)
    print("wrote", Path(out) / "trials.csv")
