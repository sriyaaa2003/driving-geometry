"""Argoverse 2 sensor dataset loader (open S3 bucket, CC BY-NC-SA 4.0, no account needed).

Only what the calibration study needs is fetched for a log: calibration, vehicle poses, a subsample of
LiDAR sweeps and, for each sweep, the nearest image of the chosen cameras.
"""
from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

BASE = "https://argoverse.s3.amazonaws.com/"
PREFIX = "datasets/av2/sensor"


def _get(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (fixed https bucket)
        tmp.rename(dest)
    return dest


def _list(prefix: str, delimiter: str | None = None) -> list[str]:
    out, token = [], None
    while True:
        q = f"{BASE}?list-type=2&prefix={prefix}&max-keys=1000"
        if delimiter:
            q += f"&delimiter={delimiter}"
        if token:
            q += "&continuation-token=" + urllib.request.quote(token, safe="")
        xml = urllib.request.urlopen(q).read().decode()  # noqa: S310
        out += re.findall(r"<Key>([^<]*)</Key>", xml) + re.findall(r"<Prefix>([^<]*)</Prefix>", xml)[1:]
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", xml)
        if not m:
            return out
        token = m.group(1).replace("&#43;", "+")


def list_logs(split: str = "val") -> list[str]:
    pre = f"{PREFIX}/{split}/"
    return sorted(p.rstrip("/").split("/")[-1] for p in _list(pre, "/") if p.startswith(pre) and p != pre)


@dataclass
class Frame:
    t_lidar: int
    lidar: np.ndarray  # (N,3) ego frame at sweep time
    images: dict[str, tuple[np.ndarray, int]]  # camera -> (uint8 gray image, timestamp)


@dataclass
class Log:
    log_id: str
    intrinsics: pd.DataFrame
    extrinsics: pd.DataFrame  # egovehicle_SE3_sensor
    poses: pd.DataFrame  # city_SE3_egovehicle
    frames: list[Frame]

    def cam_to_ego(self, cam: str) -> np.ndarray:
        r = self.extrinsics[self.extrinsics.sensor_name == cam].iloc[0]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()
        T[:3, 3] = [r.tx_m, r.ty_m, r.tz_m]
        return T

    def intr(self, cam: str) -> dict[str, float]:
        r = self.intrinsics[self.intrinsics.sensor_name == cam].iloc[0]
        return {k: float(r[k]) for k in ("fx_px", "fy_px", "cx_px", "cy_px", "k1", "k2", "k3", "width_px", "height_px")}

    def pose_at(self, t_ns: int) -> np.ndarray:
        """4x4 city_T_ego at time t (translation linear, rotation slerp)."""
        p = self.poses
        ts = p.timestamp_ns.to_numpy()
        i = int(np.clip(np.searchsorted(ts, t_ns) - 1, 0, len(ts) - 2))
        a = float(np.clip((t_ns - ts[i]) / max(ts[i + 1] - ts[i], 1), 0, 1))
        rots = Rotation.from_quat(p[["qx", "qy", "qz", "qw"]].to_numpy()[i : i + 2])
        R = Slerp([0, 1], rots)(a).as_matrix()
        tr = (1 - a) * p[["tx_m", "ty_m", "tz_m"]].to_numpy()[i] + a * p[["tx_m", "ty_m", "tz_m"]].to_numpy()[i + 1]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, tr
        return T


def load_log(log_id: str, cache: Path, cameras: list[str], split: str = "val", every: int = 3,
             max_frames: int = 60, downscale: int = 4) -> Log:
    import cv2

    root = f"{PREFIX}/{split}/{log_id}"
    d = cache / log_id
    cal = {n: pd.read_feather(_get(f"{BASE}{root}/calibration/{n}.feather", d / f"{n}.feather"))
           for n in ("egovehicle_SE3_sensor", "intrinsics")}
    poses = pd.read_feather(_get(f"{BASE}{root}/city_SE3_egovehicle.feather", d / "poses.feather"))
    lidar_keys = sorted(k for k in _list(f"{root}/sensors/lidar/") if k.endswith(".feather"))[::every][:max_frames]
    cam_keys = {c: sorted(k for k in _list(f"{root}/sensors/cameras/{c}/") if k.endswith(".jpg")) for c in cameras}
    cam_ts = {c: np.array([int(Path(k).stem) for k in ks]) for c, ks in cam_keys.items()}
    frames = []
    for k in lidar_keys:
        t = int(Path(k).stem)
        pts = pd.read_feather(_get(BASE + k, d / "lidar" / Path(k).name))[["x", "y", "z"]].to_numpy(np.float32)
        imgs = {}
        for c in cameras:
            j = int(np.argmin(np.abs(cam_ts[c] - t)))
            p = _get(BASE + cam_keys[c][j], d / c / Path(cam_keys[c][j]).name)
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, (g.shape[1] // downscale, g.shape[0] // downscale), interpolation=cv2.INTER_AREA)
            imgs[c] = (g, int(cam_ts[c][j]))
        frames.append(Frame(t, pts, imgs))
    return Log(log_id, cal["intrinsics"], cal["egovehicle_SE3_sensor"], poses, frames)
