"""Export a synthetic rig (images, intrinsics, odometry, LiDAR points, true + perturbed extrinsics) to .npz
so the GPU Gaussian-splatting calibration (gs_calib.py) can be smoke-tested against a KNOWN answer before it
is pointed at real data.

  python -m dgeo.calibration.export_synth --path curve --camera pinhole --out data/synth_curve_pinhole.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dgeo.calibration.core import perturbed_init
from dgeo.calibration.scene import make_scene


def mat(R: torch.Tensor, t: torch.Tensor) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.numpy(), t.numpy()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="curve", choices=["straight", "curve"])
    ap.add_argument("--camera", default="pinhole", choices=["pinhole", "equidistant"])
    ap.add_argument("--open-road", action="store_true")
    ap.add_argument("--rot-deg", type=float, default=1.0)
    ap.add_argument("--trans-cm", type=float, default=3.0)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = make_scene(a.path, n_frames=a.frames, walls=not a.open_road, camera_model=a.camera)
    Ri, ti = perturbed_init(s, a.rot_deg, a.trans_cm, seed=1)
    views = [(c, f) for c in range(len(s.cams)) for f in range(len(s.poses))]
    images = np.stack([s.images[v].numpy() for v in views]).astype(np.float32)  # (V,H,W)
    cam = s.cams[0]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        images=np.repeat(images[..., None], 3, axis=-1),
        view_cam=np.array([v[0] for v in views]), view_frame=np.array([v[1] for v in views]),
        K=cam.intrinsics_matrix().astype(np.float32), width=cam.width, height=cam.height, camera_model=cam.model,
        T_wl=np.stack([mat(R, t) for R, t in s.poses]), points=s.points.numpy().astype(np.float32),
        T_cl_true=np.stack([mat(R, t) for R, t in zip(s.R_cl, s.t_cl)]),
        T_cl_init=np.stack([mat(R, t) for R, t in zip(Ri, ti)]),
    )
    print(f"wrote {a.out}: {len(views)} views, {len(s.points)} points, camera={cam.model}")


if __name__ == "__main__":
    main()
