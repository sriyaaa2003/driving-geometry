"""Rotation / camera-model utilities (torch, differentiable) and pose-error metrics (numpy)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def skew(w: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(w[..., 0])
    return torch.stack(
        [
            torch.stack([z, -w[..., 2], w[..., 1]], -1),
            torch.stack([w[..., 2], z, -w[..., 0]], -1),
            torch.stack([-w[..., 1], w[..., 0], z], -1),
        ],
        -2,
    )


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula, safe (and differentiable) at w = 0."""
    th2 = (w * w).sum(-1, keepdim=True)
    th = torch.sqrt(th2 + 1e-18)
    small = th2 < 1e-12
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-18))
    k = skew(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(k.shape)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def perturb(R0: torch.Tensor, t0: torch.Tensor, xi: torch.Tensor):
    """Apply a 6-vector (rotation w [rad], translation v [m]) to a pose: R = exp(w) R0, t = t0 + v."""
    return so3_exp(xi[..., :3]) @ R0, t0 + xi[..., 3:]


def rot_z(a: float, dtype=torch.float64) -> torch.Tensor:
    c, s = float(np.cos(a)), float(np.sin(a))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)


@dataclass
class Camera:
    """Camera looking along +z, x right, y down. model: 'pinhole' or 'equidistant' (ideal fisheye, r = f * theta)."""

    model: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    max_theta: float = 1.45  # rad, valid field of view of a fisheye (half-angle)

    def project(self, pc: torch.Tensor):
        """pc (...,3) in camera coords -> (uv (...,2), valid (...) bool)."""
        x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
        if self.model == "pinhole":
            zz = z.clamp_min(1e-6)
            u = self.fx * x / zz + self.cx
            v = self.fy * y / zz + self.cy
            valid = z > 1e-3
        elif self.model == "equidistant":
            r = torch.sqrt(x * x + y * y + 1e-18)
            theta = torch.atan2(r, z)
            s = theta / r
            u = self.fx * s * x + self.cx
            v = self.fy * s * y + self.cy
            valid = theta < self.max_theta
        else:
            raise ValueError(self.model)
        valid = valid & (u >= 0) & (u <= self.width - 1) & (v >= 0) & (v <= self.height - 1)
        return torch.stack([u, v], -1), valid

    def rays(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Pixel coords -> unit rays (...,3) in camera coords."""
        dx, dy = (u - self.cx) / self.fx, (v - self.cy) / self.fy
        if self.model == "pinhole":
            d = torch.stack([dx, dy, torch.ones_like(dx)], -1)
            return d / d.norm(dim=-1, keepdim=True)
        rho = torch.sqrt(dx * dx + dy * dy + 1e-18)
        theta = rho
        s = torch.sin(theta) / rho
        return torch.stack([s * dx, s * dy, torch.cos(theta)], -1)

    def intrinsics_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


def rot_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pose_error(T_est: np.ndarray, T_gt: np.ndarray):
    """(rotation error [deg], translation error [m]) between two 4x4 poses."""
    dR = T_est[:3, :3].T @ T_gt[:3, :3]
    return rot_angle_deg(dR), float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
