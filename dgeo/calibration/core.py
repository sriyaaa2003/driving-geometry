"""Target-free rig calibration core: photometric multi-view consistency of LiDAR points, observability
analysis (Gauss-Newton information matrix) and a road-plane mounting prior.

This is the CPU-testable counterpart of the 3D-Gaussian-Splatting calibration in gs_calib.py: here the scene
appearance is "whatever colour the other views agree on" instead of a learned Gaussian field, which isolates the
geometric question -- which extrinsic directions does a drive actually constrain?

Parameters p are (C, 6) per camera in (degrees, centimetres): rotation applied as exp(w) R0, translation t0 + v.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dgeo.calibration.scene import SynthScene
from dgeo.geometry import perturb, pose_error, so3_exp

F64 = torch.float64
PARAM_NAMES = ["rx_deg", "ry_deg", "rz_deg", "tx_cm", "ty_cm", "tz_cm"]


@dataclass
class RoadPrior:
    """Gaussian prior on camera height above the road and on pitch/roll relative to the road plane
    (mounting tolerances known from the vehicle drawing). It constrains 3 of 6 DoF per camera."""

    sigma_h_m: float = 0.03
    sigma_angle_deg: float = 0.5
    weight: float = 1.0


def _ground_frame_values(R_cl: torch.Tensor, t_cl: torch.Tensor, ground_z: float):
    """(height above ground [m], pitch [rad], roll [rad]) of a camera, from its camera-from-lidar pose."""
    pos = -(R_cl.transpose(-1, -2) @ t_cl[..., None])[..., 0]
    height = pos[..., 2] - ground_z
    axes = R_cl.transpose(-1, -2)  # columns = camera axes in lidar frame
    pitch = torch.asin(axes[..., 2, 2].clamp(-1, 1))  # optical axis z-component
    roll = torch.asin(axes[..., 2, 0].clamp(-1, 1))  # camera x axis z-component
    return height, pitch, roll


class CalibProblem:
    def __init__(self, scene: SynthScene, init_R, init_t, sigma_c: float = 0.02, prior: RoadPrior | None = None,
                 prior_nominal=None, levels: int = 3, max_range: float = 30.0):
        self.s = scene
        self.C = len(scene.cams)
        self.R0 = torch.stack(init_R)
        self.t0 = torch.stack(init_t)
        self.sigma_c = sigma_c
        self.max_range = max_range
        self.prior = prior
        # prior is centred on the mounting values (default: those of the true extrinsics)
        Rn = torch.stack(prior_nominal[0]) if prior_nominal else torch.stack(scene.R_cl)
        tn = torch.stack(prior_nominal[1]) if prior_nominal else torch.stack(scene.t_cl)
        self.prior_nom = torch.stack(_ground_frame_values(Rn, tn, scene.ground_z), -1)  # (C,3)
        self.views = [(c, f) for c in range(self.C) for f in range(len(scene.poses))]
        self.pyr = []
        for lvl in range(levels):
            d = {}
            for k, img in scene.images.items():
                x = img[None, None]
                for _ in range(lvl):
                    x = F.avg_pool2d(x, 2)
                d[k] = x
            self.pyr.append(d)

    # ---- model -------------------------------------------------------------------------------------------
    def extrinsics(self, p: torch.Tensor):
        P = p.view(self.C, 6)
        return perturb(self.R0, self.t0, torch.cat([P[:, :3] * (np.pi / 180.0), P[:, 3:] / 100.0], 1))

    def _colors(self, R, t, level: int):
        pts = self.s.points
        cols, valid = [], []
        for c, f in self.views:
            R_wl, t_wl = self.s.poses[f]
            p_l = (pts - t_wl) @ R_wl  # = R_wl^T (p - t)
            p_c = p_l @ R[c].T + t[c]
            cam = self.s.cams[c]
            uv, ok = cam.project(p_c)
            ok = ok & (p_c.norm(dim=-1) < self.max_range)
            # pixel centres sit at integer coords; align_corners=False keeps every pyramid level consistent
            gx = 2 * (uv[:, 0] + 0.5) / cam.width - 1
            gy = 2 * (uv[:, 1] + 0.5) / cam.height - 1
            grid = torch.stack([gx, gy], -1)[None, :, None, :]
            col = F.grid_sample(self.pyr[level][(c, f)], grid, mode="bilinear", padding_mode="border",
                                align_corners=False)[0, 0, :, 0]
            cols.append(col)
            valid.append(ok)
        return torch.stack(cols), torch.stack(valid)

    def make_mask(self, p: torch.Tensor, level: int = 0) -> torch.Tensor:
        with torch.no_grad():
            R, t = self.extrinsics(p)
            _, ok = self._colors(R, t, level)
            return ok & (ok.sum(0, keepdim=True) >= 2)

    def residual(self, p: torch.Tensor, level: int = 0, mask: torch.Tensor | None = None) -> torch.Tensor:
        R, t = self.extrinsics(p)
        cols, ok = self._colors(R, t, level)
        m = mask if mask is not None else (ok & (ok.sum(0, keepdim=True) >= 2))
        mf = m.to(F64)
        mean = (cols * mf).sum(0, keepdim=True) / mf.sum(0, keepdim=True).clamp_min(1.0)
        r = ((cols - mean) * mf).reshape(-1) / self.sigma_c
        if self.prior is not None and self.prior.weight > 0:
            h, pit, rol = _ground_frame_values(R, t, self.s.ground_z)
            cur = torch.stack([h, pit, rol], -1)
            d = cur - self.prior_nom
            scale = torch.tensor([1 / self.prior.sigma_h_m, 1 / np.radians(self.prior.sigma_angle_deg),
                                  1 / np.radians(self.prior.sigma_angle_deg)], dtype=F64)
            r = torch.cat([r, (d * scale).reshape(-1) * float(np.sqrt(self.prior.weight))])
        return r

    def errors(self, p: torch.Tensor):
        """Per-camera (rotation deg, translation cm) error of the current extrinsics against ground truth."""
        R, t = self.extrinsics(p)
        out = []
        for c in range(self.C):
            T = np.eye(4); T[:3, :3] = R[c].detach().numpy(); T[:3, 3] = t[c].detach().numpy()
            G = np.eye(4); G[:3, :3] = self.s.R_cl[c].numpy(); G[:3, 3] = self.s.t_cl[c].numpy()
            rd, tm = pose_error(T, G)
            out.append((rd, tm * 100.0))
        return out


def _position_error_lidar(self, p: torch.Tensor) -> np.ndarray:
    """(C,3) camera-position error in the LiDAR frame [cm]: estimated minus true."""
    R, t = self.extrinsics(p)
    est = -(R.transpose(-1, -2) @ t[..., None])[..., 0]
    true = -(torch.stack(self.s.R_cl).transpose(-1, -2) @ torch.stack(self.s.t_cl)[..., None])[..., 0]
    return ((est - true) * 100.0).detach().numpy()


CalibProblem.position_error_lidar = _position_error_lidar


def fd_jacobian(fun, p: torch.Tensor, eps: float = 0.02) -> torch.Tensor:
    cols = []
    for i in range(p.numel()):
        e = torch.zeros_like(p)
        e[i] = eps
        cols.append((fun(p + e) - fun(p - e)) / (2 * eps))
    return torch.stack(cols, 1)


def observability(problem: CalibProblem, level: int = 0) -> dict:
    """Gauss-Newton information J^T J at the true calibration (p = 0 when init == truth).
    Returns Cramer-Rao standard deviations per parameter, eigen-spectrum, condition number and the
    weakest direction, all in (deg, cm)."""
    p = torch.zeros(problem.C * 6, dtype=F64)
    mask = problem.make_mask(p, level)
    J = fd_jacobian(lambda q: problem.residual(q, level, mask), p)
    info = (J.T @ J).numpy()
    n = info.shape[0]
    evals, evecs = np.linalg.eigh(info + 1e-9 * np.eye(n))
    cov = np.linalg.inv(info + 1e-9 * np.eye(n))
    crb = np.sqrt(np.clip(np.diag(cov), 0, None)).reshape(problem.C, 6)
    w = evecs[:, 0]
    top = np.argsort(-np.abs(w))[:3]
    # common-mode shift of ALL cameras along a LiDAR-frame axis (the direction photometric consistency
    # cannot see when it is tangent to every surface): marginal std in cm per unit shift
    common = {}
    Rn = torch.stack(problem.R0).numpy() if isinstance(problem.R0, list) else problem.R0.numpy()
    for ai, axis in enumerate("xyz"):
        e = np.zeros(3); e[ai] = 1.0
        v = np.zeros(n)
        for c in range(problem.C):
            v[c * 6 + 3:c * 6 + 6] = -Rn[c] @ e  # camera-frame translation that moves camera c by 1 cm along `axis`
        common[axis] = float(np.sqrt(max(v @ cov @ v, 0.0) / problem.C))
    return dict(
        crb_std=crb, eigenvalues=evals, cond=float(evals[-1] / max(evals[0], 1e-30)),
        weakest_direction=[(f"cam{i // 6}.{PARAM_NAMES[i % 6]}", float(w[i])) for i in top],
        common_mode_std_cm=common,
        n_points_used=int(mask.any(0).sum()),
    )


def solve(problem: CalibProblem, levels=(2, 1, 0), iters: int = 12, p0=None, verbose: bool = False):
    """Coarse-to-fine Levenberg-Marquardt on the photometric-consistency (+ prior) objective."""
    p = torch.zeros(problem.C * 6, dtype=F64) if p0 is None else p0.clone()
    for lvl in levels:
        mask = problem.make_mask(p, lvl)
        fun = lambda q: problem.residual(q, lvl, mask)
        mu = 1e-2
        r = fun(p)
        loss = float((r * r).sum())
        for it in range(iters):
            J = fd_jacobian(fun, p)
            H, g = J.T @ J, J.T @ r
            while mu < 1e8:
                step = torch.linalg.solve(H + mu * torch.diag(torch.diag(H)) + 1e-6 * torch.eye(len(p), dtype=F64), -g)
                step = step.clamp(-3.0, 3.0)  # trust region: <= 3 deg / 3 cm per iteration
                r_new = fun(p + step)
                l_new = float((r_new * r_new).sum())
                if l_new < loss:
                    p, r, loss, mu = p + step, r_new, l_new, max(mu / 3, 1e-6)
                    break
                mu *= 4
            else:
                break
        if verbose:
            print(f"level {lvl}: loss {loss:.3f} errors {problem.errors(p)}")
    return p


def perturbed_init(scene: SynthScene, rot_deg: float, trans_cm: float, seed: int = 0):
    """Truth + a random rotation of `rot_deg` and translation of `trans_cm` (random directions) per camera."""
    rng = np.random.default_rng(seed)
    Rs, ts = [], []
    for R, t in zip(scene.R_cl, scene.t_cl):
        a = rng.normal(size=3); a = a / np.linalg.norm(a) * np.radians(rot_deg)
        b = rng.normal(size=3); b = b / np.linalg.norm(b) * trans_cm / 100.0
        Rs.append(so3_exp(torch.tensor(a, dtype=F64)) @ R)
        ts.append(t + torch.tensor(b, dtype=F64))
    return Rs, ts
