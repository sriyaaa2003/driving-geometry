"""Which extrinsic directions does a real drive constrain when calibrating a camera from the scene?

Setup (Argoverse 2 ring cameras + LiDAR): camera A (front centre) is trusted. For camera B (front left or
front right) the dataset's own extrinsic is taken as ground truth. LiDAR points seen by both cameras are
projected into both images. The photometric difference between the two views of the same 3D points is
minimised over a 6-DoF correction of camera B's extrinsic, starting from a random perturbation.
Reported per axis (ego frame: x forward, y left, z up): the error left after the solve and the
Cramer-Rao standard deviation from the Jacobian of the residuals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

AXES = ["x (forward)", "y (left)", "z (up)"]


def lcn(img: torch.Tensor, win: int = 9) -> torch.Tensor:
    """Local contrast normalisation of a (1,1,H,W) image so two cameras with different exposure agree."""
    pad = win // 2
    mu = F.avg_pool2d(img, win, 1, pad, count_include_pad=False)
    var = F.avg_pool2d(img * img, win, 1, pad, count_include_pad=False) - mu * mu
    return (img - mu) / (var.clamp_min(0).sqrt() + 8.0)


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    th = w.norm() + 1e-12
    k = w / th
    K = torch.zeros(3, 3, dtype=w.dtype, device=w.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, dtype=w.dtype, device=w.device) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def se3_exp(theta: torch.Tensor) -> torch.Tensor:
    """First-order-exact small correction: rotation by so3_exp(w), translation v (kept separate)."""
    T = torch.eye(4, dtype=theta.dtype, device=theta.device)
    T = T.clone()
    T[:3, :3] = so3_exp(theta[:3])
    T[:3, 3] = theta[3:]
    return T


def project(P_cam: torch.Tensor, intr: dict[str, float], scale: float):
    """Pinhole + radial distortion. Returns pixel coords (N,2) at the image's downscaled size and depth."""
    z = P_cam[:, 2]
    x, y = P_cam[:, 0] / z.clamp_min(1e-6), P_cam[:, 1] / z.clamp_min(1e-6)
    r2 = x * x + y * y
    d = 1 + intr["k1"] * r2 + intr["k2"] * r2**2 + intr["k3"] * r2**3
    u = (intr["fx_px"] * x * d + intr["cx_px"]) * scale
    v = (intr["fy_px"] * y * d + intr["cy_px"]) * scale
    return torch.stack([u, v], 1), z


def sample(img: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    H, W = img.shape[-2:]
    g = torch.stack([uv[:, 0] / (W - 1) * 2 - 1, uv[:, 1] / (H - 1) * 2 - 1], 1)[None, None]
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0]


@dataclass
class FrameData:
    P_a: torch.Tensor  # points in ego frame at camera A's time (N,3)
    P_b: torch.Tensor  # same points in ego frame at camera B's time
    ia: dict[float, torch.Tensor]  # scale -> normalised intensity of A at the projected pixels (N,)
    img_b: dict[float, torch.Tensor]  # scale -> normalised image of B (1,1,H,W)


class Pair:
    """Frames of one drive segment for the camera pair (A fixed, B estimated)."""

    def __init__(self, log, frames, cam_a: str, cam_b: str, device: str, down: int, max_pts: int = 20000,
                 seed: int = 0, zmin: float = 4.0, zmax: float = 50.0):
        self.log, self.cam_a, self.cam_b, self.device = log, cam_a, cam_b, device
        self.down = down
        self.T_a = torch.tensor(log.cam_to_ego(cam_a), dtype=torch.float32, device=device)
        self.T_b = torch.tensor(log.cam_to_ego(cam_b), dtype=torch.float32, device=device)
        self.ia_cfg, self.ib_cfg = log.intr(cam_a), log.intr(cam_b)
        rng = np.random.default_rng(seed)
        self.data: list[FrameData] = []
        for fr in frames:
            ga, ta = fr.images[cam_a]
            gb, tb = fr.images[cam_b]
            Pl = fr.lidar
            c_l = log.pose_at(fr.t_lidar)
            to_a = np.linalg.inv(log.pose_at(ta)) @ c_l
            to_b = np.linalg.inv(log.pose_at(tb)) @ c_l
            P = np.concatenate([Pl, np.ones((len(Pl), 1), np.float32)], 1)
            Pa = torch.tensor((P @ to_a.T)[:, :3], dtype=torch.float32, device=device)
            Pb = torch.tensor((P @ to_b.T)[:, :3], dtype=torch.float32, device=device)
            imgs = {}
            for cam, g in ((cam_a, ga), (cam_b, gb)):
                t = torch.tensor(g, dtype=torch.float32, device=device)[None, None]
                imgs[cam] = {1.0: lcn(t), 0.5: lcn(F.avg_pool2d(t, 2))}
            ua, za = project((Pa @ torch.linalg.inv(self.T_a)[:3, :3].T) + torch.linalg.inv(self.T_a)[:3, 3],
                             self.ia_cfg, 1 / down)
            ub, zb = project((Pb @ torch.linalg.inv(self.T_b)[:3, :3].T) + torch.linalg.inv(self.T_b)[:3, 3],
                             self.ib_cfg, 1 / down)
            Ha, Wa = ga.shape
            Hb, Wb = gb.shape
            m = ((za > zmin) & (za < zmax) & (zb > zmin) & (zb < zmax) & (ua[:, 0] > 8) & (ua[:, 0] < Wa - 8)
                 & (ua[:, 1] > 8) & (ua[:, 1] < Ha - 8) & (ub[:, 0] > 8) & (ub[:, 0] < Wb - 8)
                 & (ub[:, 1] > 8) & (ub[:, 1] < Hb - 8))
            idx = torch.nonzero(m)[:, 0]
            # keep points that are textured in camera A (chosen without looking at B or at the solve):
            # flat regions match under any shift and would pull the optimum away from the truth
            ia_full = sample(imgs[cam_a][1.0], ua[idx])
            idx = idx[ia_full.abs() > 0.5]
            if len(idx) > max_pts:
                idx = idx[torch.tensor(rng.choice(len(idx), max_pts, replace=False), device=device)]
            if len(idx) < 200:
                continue
            ia = {sc: sample(imgs[cam_a][sc], ua[idx] * sc) for sc in (1.0, 0.5)}
            self.data.append(FrameData(Pa[idx], Pb[idx], ia, {1.0: imgs[cam_b][1.0], 0.5: imgs[cam_b][0.5]}))

    def n_points(self) -> int:
        return sum(len(d.P_a) for d in self.data)

    def residuals(self, theta: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        T_est = self.T_b @ se3_exp(theta)
        Tinv = torch.linalg.inv(T_est)
        out = []
        for d in self.data:
            Pc = d.P_b @ Tinv[:3, :3].T + Tinv[:3, 3]
            uv, _ = project(Pc, self.ib_cfg, scale / self.down)
            out.append(d.ia[scale] - sample(d.img_b[scale], uv))
        return torch.cat(out)

    def loss(self, theta: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        r = self.residuals(theta, scale)
        return torch.sqrt(r * r + 0.05**2).mean()  # Charbonnier

    def solve(self, theta0: torch.Tensor, steps: int = 150) -> torch.Tensor:
        theta = theta0.clone().requires_grad_(True)
        for scale, lr_scale in ((0.5, 1.0), (1.0, 0.3)):
            opt = torch.optim.Adam([{"params": [theta]}], lr=1e-3 * lr_scale)
            for _ in range(steps):
                opt.zero_grad()
                self.loss(theta, scale).backward()
                # rotation (rad) and translation (m) have different natural scales
                with torch.no_grad():
                    theta.grad[3:] *= 10.0
                opt.step()
        return theta.detach()

    def crb(self) -> tuple[np.ndarray, float]:
        """Cramer-Rao covariance (6x6) of the correction at the true extrinsic, from a finite-difference
        Jacobian of the residuals: sigma^2 * (J^T J)^-1."""
        dev = self.device
        steps = [1e-3] * 3 + [1e-2] * 3
        cols = []
        with torch.no_grad():
            for i, h in enumerate(steps):
                e = torch.zeros(6, device=dev)
                e[i] = h
                cols.append((self.residuals(e) - self.residuals(-e)) / (2 * h))
            r = self.residuals(torch.zeros(6, device=dev))
        J = torch.stack(cols, 1)
        sigma = float(r.pow(2).mean().sqrt())
        H = (J.T @ J).double().cpu().numpy()
        return np.linalg.pinv(H) * sigma**2, sigma


def to_ego_axes(T_b_ego: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(rot x,y,z deg, trans x,y,z cm) of a correction expressed in ego axes."""
    R = T_b_ego[:3, :3]
    return np.concatenate([np.degrees(R @ theta[:3]), 100 * (R @ theta[3:])])


def segment_stats(log, frames) -> dict[str, float]:
    p0, p1 = log.pose_at(frames[0].t_lidar), log.pose_at(frames[-1].t_lidar)
    yaw = Rotation.from_matrix((np.linalg.inv(p0) @ p1)[:3, :3]).as_euler("zyx", degrees=True)[0]
    return {"heading_change_deg": float(abs(yaw)), "path_m": float(np.linalg.norm(p1[:3, 3] - p0[:3, 3]))}


def run_segment(log, frames, cam_a, cam_b, device, down, n_trials, seed, rot_deg=0.5, trans_cm=5.0,
                zmin=4.0, zmax=50.0, start_at_truth=False, tag="main"):
    pair = Pair(log, frames, cam_a, cam_b, device, down, seed=seed, zmin=zmin, zmax=zmax)
    if len(pair.data) < max(3, len(frames) // 2):
        return []
    rng = np.random.default_rng(seed)
    cov, sigma = pair.crb()
    T_b = log.cam_to_ego(cam_b)
    R = T_b[:3, :3]
    M = np.zeros((6, 6))
    M[:3, :3] = np.degrees(1) * R
    M[3:, 3:] = 100 * R
    crb_ego = np.sqrt(np.clip(np.diag(M @ cov @ M.T), 0, None))  # std per ego axis (deg, cm)
    rows = []
    with torch.no_grad():
        l_true = float(pair.loss(torch.zeros(6, device=device)))
    for t in range(n_trials):
        th0 = np.concatenate([np.radians(rng.normal(0, rot_deg, 3)), rng.normal(0, trans_cm / 100, 3)])
        if start_at_truth:
            th0 = np.zeros(6)
        th = pair.solve(torch.tensor(th0, dtype=torch.float32, device=device)).cpu().numpy()
        err0, err = to_ego_axes(T_b, th0), to_ego_axes(T_b, th)
        rows.append(dict(experiment=tag, method="photometric", cam_b=cam_b, trial=t, zmin=zmin, zmax=zmax,
                         rot_sigma_deg=rot_deg, trans_sigma_cm=trans_cm,
                         n_points=pair.n_points(), n_frames=len(pair.data), sigma=sigma,
                         loss_true=l_true, **{f"init_{k}": v for k, v in zip(_names(), err0)},
                         **{f"err_{k}": v for k, v in zip(_names(), err)},
                         **{f"crb_{k}": v for k, v in zip(_names(), crb_ego)}, **segment_stats(log, frames)))
    return rows


def _names() -> list[str]:
    return [f"rot_{a}_deg" for a in "xyz"] + [f"trans_{a}_cm" for a in "xyz"]
