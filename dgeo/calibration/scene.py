"""Synthetic driving scene with a multi-camera fisheye rig and a LiDAR, with exact ground truth.

Frames: world = LiDAR frame at t=0 (x forward, y left, z up). Cameras look along +z with x right, y down.
Images are ray-cast analytically from textured planes (ground + side walls [+ front wall]), independent of the
LiDAR points, so calibration experiments have a known optimum.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from dgeo.geometry import Camera, rot_z, so3_exp

F64 = torch.float64
R_BASE = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=F64)  # lidar axes -> camera axes


def rot_x(a: float) -> torch.Tensor:
    c, s = float(np.cos(a)), float(np.sin(a))
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=F64)


def cam_extrinsic(yaw_deg: float, pos, pitch_deg: float = 0.0):
    """Camera pose in the LiDAR frame -> (R_cl, t_cl): camera-from-lidar rotation/translation."""
    R = rot_x(np.radians(pitch_deg)) @ R_BASE @ rot_z(-np.radians(yaw_deg))
    pos = torch.tensor(pos, dtype=F64)
    return R, -R @ pos


class Texture:
    """Smooth multi-frequency 3-D texture, value in [0,1]; stands in for scene appearance."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        mags = np.concatenate([rng.uniform(0.3, 1.0, 4), rng.uniform(0.9, 2.2, 6), rng.uniform(3.0, 7.0, 8)])
        dirs = rng.normal(size=(len(mags), 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        self.k = torch.tensor(dirs * mags[:, None], dtype=F64)
        self.phase = torch.tensor(rng.uniform(0, 2 * np.pi, len(mags)), dtype=F64)
        self.amp = torch.tensor(rng.uniform(0.5, 1.0, len(mags)), dtype=F64)

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        s = (torch.sin(p @ self.k.T + self.phase) * self.amp).sum(-1) / self.amp.sum()
        return 0.5 + 0.5 * s


@dataclass
class SynthScene:
    cams: list
    R_cl: list
    t_cl: list
    poses: list  # [(R_wl, t_wl)] per frame
    images: dict  # (cam, frame) -> (H, W) float64 tensor
    points: torch.Tensor  # (N,3) in world
    ground_z: float
    tex: Texture
    name: str = ""
    meta: dict = field(default_factory=dict)


def render(cam: Camera, R_wc: torch.Tensor, t_wc: torch.Tensor, planes, tex: Texture, ss: int = 3) -> torch.Tensor:
    """Ray-cast with ss x ss supersampling per pixel (area-averaged, like a real camera)."""
    v0, u0 = torch.meshgrid(torch.arange(cam.height, dtype=F64), torch.arange(cam.width, dtype=F64), indexing="ij")
    acc = torch.zeros_like(u0)
    for i in range(ss):
        for j in range(ss):
            u = u0 + (i + 0.5) / ss - 0.5
            v = v0 + (j + 0.5) / ss - 0.5
            d = cam.rays(u, v) @ R_wc.T
            best = torch.full(u.shape, float("inf"), dtype=F64)
            for axis, off in planes:
                denom = d[..., axis]
                tt = (off - t_wc[axis]) / torch.where(denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom)
                ok = (tt > 0.1) & (tt < 150.0) & (tt < best)
                best = torch.where(ok, tt, best)
            hit = t_wc + best[..., None] * d
            img = tex(torch.where(torch.isfinite(best)[..., None], hit, torch.zeros_like(hit)))
            acc = acc + torch.where(torch.isfinite(best), img, torch.full_like(img, 0.5))
    return acc / (ss * ss)


def make_trajectory(path: str, n_frames: int, step: float):
    poses, yaw, pos = [], 0.0, np.zeros(2)
    for _ in range(n_frames):
        poses.append((rot_z(yaw), torch.tensor([pos[0], pos[1], 0.0], dtype=F64)))
        if path == "curve":
            yaw += np.radians(6.0)
        pos = pos + step * np.array([np.cos(yaw), np.sin(yaw)])
    return poses


def make_scene(path: str = "straight", n_frames: int = 4, step: float = 2.0, n_points: int = 3000,
               size=(192, 240), focal: float = 82.0, seed: int = 0, front_wall: bool = False,
               walls: bool = True, lidar_noise: float = 0.0, camera_model: str = "equidistant") -> SynthScene:
    """Rig: front + left + right cameras (ideal equidistant fisheye by default), ~0.3 m below the LiDAR."""
    H, W = size
    if camera_model == "pinhole":
        focal = 100.0
    cam = Camera(camera_model, focal, focal, (W - 1) / 2, (H - 1) / 2, W, H)
    rig = [(0.0, (1.2, 0.0, -0.30)), (90.0, (0.4, 0.9, -0.30)), (-90.0, (0.4, -0.9, -0.30))]
    R_cl, t_cl = [], []
    for yaw, pos in rig:
        R, t = cam_extrinsic(yaw, pos)
        R_cl.append(R)
        t_cl.append(t)
    ground_z, wall_y = -1.8, 7.0
    planes = [(2, ground_z)] + ([(1, wall_y), (1, -wall_y)] if walls else []) + ([(0, 60.0)] if front_wall else [])
    tex = Texture(seed)
    poses = make_trajectory(path, n_frames, step)
    images = {}
    for f, (R_wl, t_wl) in enumerate(poses):
        for c in range(len(rig)):
            cam_pos_l = -R_cl[c].T @ t_cl[c]
            R_wc = R_wl @ R_cl[c].T
            t_wc = t_wl + R_wl @ cam_pos_l
            images[(c, f)] = render(cam, R_wc, t_wc, planes, tex)
    rng = np.random.default_rng(seed + 1)
    n3 = n_points // 3 if walls else n_points
    reach = step * (n_frames - 1) + 40.0
    g = np.stack([rng.uniform(-6, reach, n3), rng.uniform(-wall_y, wall_y, n3), np.full(n3, ground_z)], 1)
    wl = np.stack([rng.uniform(-6, reach, n3), np.full(n3, wall_y), rng.uniform(ground_z, 8.0, n3)], 1)
    wr = np.stack([rng.uniform(-6, reach, n3), np.full(n3, -wall_y), rng.uniform(ground_z, 8.0, n3)], 1)
    pts = [g, wl, wr] if walls else [g]
    if front_wall:
        n4 = max(n_points // 4, 1)
        pts.append(np.stack([np.full(n4, 60.0), rng.uniform(-wall_y, wall_y, n4), rng.uniform(ground_z, 8.0, n4)], 1))
    pts = np.concatenate(pts)
    if lidar_noise > 0:
        pts = pts + rng.normal(0, lidar_noise, pts.shape)
    return SynthScene([cam] * len(rig), R_cl, t_cl, poses, images, torch.tensor(pts, dtype=F64), ground_z, tex,
                      name=path, meta=dict(front_wall=front_wall, walls=walls, step=step))
