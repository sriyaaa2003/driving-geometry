"""Classical baseline for the same task: 2D-3D pose estimation (PnP) from feature matches.

For every frame, SIFT features are matched between camera A and camera B. A keypoint in A that lies next to
a projected LiDAR point gets that LiDAR point's 3D position, so each match becomes a 3D point (ego frame at
B's time) paired with a pixel in B. All frames of the segment are pooled and solvePnPRansac gives camera B's
pose directly, with no starting guess. This is the standard feature-based alternative to the photometric
solver in study.py.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from dgeo.calibration.study import segment_stats, to_ego_axes, _names


def _cam_matrix(intr, down):
    return np.array([[intr["fx_px"] / down, 0, intr["cx_px"] / down],
                     [0, intr["fy_px"] / down, intr["cy_px"] / down], [0, 0, 1.0]]), \
        np.array([intr["k1"], intr["k2"], 0.0, 0.0, intr["k3"]])


def _project(P_cam, K, dist):
    uv, _ = cv2.projectPoints(P_cam.reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), K, dist)
    return uv[:, 0]


def pnp_baseline(log, frames, cam_a, cam_b, down, zmin=4.0, zmax=50.0, px_tol=2.0, ratio=0.8):
    T_a, T_b = log.cam_to_ego(cam_a), log.cam_to_ego(cam_b)
    Ka, da = _cam_matrix(log.intr(cam_a), down)
    Kb, db = _cam_matrix(log.intr(cam_b), down)
    sift = cv2.SIFT_create(nfeatures=3000)
    bf = cv2.BFMatcher()
    pts3, pts2 = [], []
    for fr in frames:
        ga, ta = fr.images[cam_a]
        gb, tb = fr.images[cam_b]
        c_l = log.pose_at(fr.t_lidar)
        P = np.concatenate([fr.lidar, np.ones((len(fr.lidar), 1), np.float32)], 1)
        Pa = (P @ (np.linalg.inv(log.pose_at(ta)) @ c_l).T)[:, :3]
        Pb = (P @ (np.linalg.inv(log.pose_at(tb)) @ c_l).T)[:, :3]
        Pca = (Pa - T_a[:3, 3]) @ T_a[:3, :3]  # ego -> camera A
        ok = (Pca[:, 2] > zmin) & (Pca[:, 2] < zmax)
        if ok.sum() < 200:
            continue
        uva = _project(Pca[ok], Ka, da)
        inside = (uva[:, 0] > 0) & (uva[:, 0] < ga.shape[1]) & (uva[:, 1] > 0) & (uva[:, 1] < ga.shape[0])
        uva, P3 = uva[inside], Pb[ok][inside]
        if len(uva) < 100:
            continue
        ka, fa = sift.detectAndCompute(ga, None)
        kb, fb = sift.detectAndCompute(gb, None)
        if fa is None or fb is None or len(ka) < 8 or len(kb) < 8:
            continue
        pa = np.array([k.pt for k in ka])
        dist, nn = cKDTree(uva).query(pa, distance_upper_bound=px_tol)
        has3d = np.isfinite(dist)
        for m, n in bf.knnMatch(fa, fb, k=2):
            if m.distance < ratio * n.distance and has3d[m.queryIdx]:
                pts3.append(P3[nn[m.queryIdx]])
                pts2.append(kb[m.trainIdx].pt)
    base = dict(experiment="baseline", method="pnp", cam_b=cam_b, trial=0, **segment_stats(log, frames),
                n_points=len(pts3))
    if len(pts3) < 30:
        return dict(base, solved=False)
    p3, p2 = np.array(pts3, np.float64), np.array(pts2, np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(p3, p2, Kb, db, iterationsCount=3000, reprojectionError=3.0,
                                             confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 20:
        return dict(base, solved=False)
    inl = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(p3[inl], p2[inl], Kb, db, rvec, tvec)
    R_cam, t_cam = cv2.Rodrigues(rvec)[0], tvec[:, 0]  # X_cam = R X_ego + t
    R_est = R_cam.T
    t_est = -R_cam.T @ t_cam  # camera B position in ego
    R_gt, t_gt = T_b[:3, :3], T_b[:3, 3]
    w = cv2.Rodrigues(R_gt.T @ R_est)[0][:, 0]
    v = R_gt.T @ (t_est - t_gt)
    err = to_ego_axes(T_b, np.concatenate([w, v]))
    return dict(base, solved=True, n_inliers=int(len(inl)), **{f"err_{k}": x for k, x in zip(_names(), err)})
