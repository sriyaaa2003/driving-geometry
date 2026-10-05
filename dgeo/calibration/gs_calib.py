"""Target-free rig calibration with 3D Gaussian Splatting (needs a CUDA GPU and `pip install gsplat`).

STATUS: written but NOT yet run on a GPU. Run `python scripts/lab_check.py` first, then the synthetic smoke test:

  python -m dgeo.calibration.export_synth --path curve --camera pinhole --out data/synth_curve_pinhole.npz
  python -m dgeo.calibration.gs_calib --data data/synth_curve_pinhole.npz --out results/gs_calib/curve_pinhole
  # control: extrinsics frozen at truth -> shows the photometric floor of the Gaussian fit
  python -m dgeo.calibration.gs_calib --data data/synth_curve_pinhole.npz --freeze-extrinsics --out results/gs_calib/control

Method: Gaussians are initialised at the LiDAR points (positions held fixed unless --learn-means, so LiDAR is the
geometric anchor); colours/opacities/scales are learned from all cameras and frames while per-camera SE(3)
corrections are optimised after a warm-up. Camera model 'fisheye' uses gsplat's fisheye rasteriser (check that its
projection matches your lens model before trusting results on real fisheye data).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dgeo.geometry import pose_error, so3_exp


def build_T(R0: torch.Tensor, t0: torch.Tensor, w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(C,4,4) camera-from-lidar after applying rotation correction exp(w) and translation correction v."""
    T = torch.eye(4, device=R0.device, dtype=R0.dtype).repeat(R0.shape[0], 1, 1)
    T[:, :3, :3] = so3_exp(w) @ R0
    T[:, :3, 3] = t0 + v
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/gs_calib/run")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=800, help="steps with extrinsics frozen")
    ap.add_argument("--lr-ext", type=float, default=3e-4)
    ap.add_argument("--freeze-extrinsics", action="store_true")
    ap.add_argument("--learn-means", action="store_true")
    ap.add_argument("--init-scale", type=float, default=0.35)
    ap.add_argument("--camera", default=None, help="override camera model: pinhole | fisheye")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found: this script needs one (gsplat is CUDA-only).")
    from gsplat import rasterization

    torch.manual_seed(a.seed)
    dev = "cuda"
    d = np.load(a.data, allow_pickle=True)
    imgs = torch.tensor(d["images"], device=dev)  # (V,H,W,3)
    view_cam, view_frame = d["view_cam"], d["view_frame"]
    H, W = int(d["height"]), int(d["width"])
    K = torch.tensor(d["K"], device=dev).float()
    cam_model = a.camera or ("fisheye" if str(d["camera_model"]) == "equidistant" else "pinhole")
    T_wl = torch.tensor(d["T_wl"], device=dev).float()
    T_lw = torch.linalg.inv(T_wl)
    pts = torch.tensor(d["points"], device=dev).float()
    T_true = torch.tensor(d["T_cl_true"], device=dev).float()
    T_init = torch.tensor(d["T_cl_init"], device=dev).float()
    C = T_init.shape[0]
    R0, t0 = T_init[:, :3, :3], T_init[:, :3, 3]

    N = len(pts)
    means = torch.nn.Parameter(pts.clone(), requires_grad=a.learn_means)
    log_scales = torch.nn.Parameter(torch.full((N, 3), float(np.log(a.init_scale)), device=dev))
    opac = torch.nn.Parameter(torch.full((N,), 1.0, device=dev))  # sigmoid(1.0) ~ 0.73
    colors = torch.nn.Parameter(torch.zeros(N, 3, device=dev))
    quats = torch.zeros(N, 4, device=dev)
    quats[:, 0] = 1.0
    ext_w = torch.nn.Parameter(torch.zeros(C, 3, device=dev))
    ext_v = torch.nn.Parameter(torch.zeros(C, 3, device=dev))

    opt_g = torch.optim.Adam(
        [{"params": [log_scales], "lr": 5e-3}, {"params": [opac], "lr": 5e-2}, {"params": [colors], "lr": 5e-2}]
        + ([{"params": [means], "lr": 1e-3}] if a.learn_means else []))
    opt_e = torch.optim.Adam([ext_w, ext_v], lr=a.lr_ext)
    bg = torch.full((1, 3), 0.5, device=dev)

    def errors():
        T = build_T(R0, t0, ext_w.detach(), ext_v.detach()).cpu().numpy()
        Tt = T_true.cpu().numpy()
        return [pose_error(T[c], Tt[c]) for c in range(C)]

    e0 = errors()
    log = []
    t_start = time.time()
    for step in range(a.steps):
        v = int(np.random.randint(len(imgs)))
        c, f = int(view_cam[v]), int(view_frame[v])
        T_cl = build_T(R0, t0, ext_w, ext_v)[c]
        viewmat = (T_cl @ T_lw[f])[None]
        out, alpha, _ = rasterization(
            means, quats, torch.exp(log_scales), torch.sigmoid(opac), torch.sigmoid(colors),
            viewmat, K[None], W, H, camera_model=cam_model, backgrounds=bg)
        loss = (out[0] - imgs[v]).abs().mean()
        opt_g.zero_grad(); opt_e.zero_grad()
        loss.backward()
        opt_g.step()
        if step >= a.warmup and not a.freeze_extrinsics:
            opt_e.step()
        if step % 200 == 0 or step == a.steps - 1:
            err = errors()
            log.append(dict(step=step, loss=float(loss), rot_deg=[e[0] for e in err], trans_cm=[e[1] * 100 for e in err]))
            print(f"step {step:5d} loss {float(loss):.4f} rot(deg) {[round(e[0], 3) for e in err]} "
                  f"trans(cm) {[round(e[1] * 100, 1) for e in err]}", flush=True)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(args=vars(a), camera_model=cam_model, seconds=time.time() - t_start,
                   start=dict(rot_deg=[e[0] for e in e0], trans_cm=[e[1] * 100 for e in e0]),
                   final=log[-1], log=log)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
