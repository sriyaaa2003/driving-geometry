"""Geometry-drift metrics: does an edited image keep the geometry of the original?

For a same-viewpoint edit (weather, lighting) the ideal answer is "nothing moved". Every metric below is
computed between ORIGINAL and EDITED image and is 0 / 1 for perfectly preserved geometry:

  dense_disp_px    median optical-flow magnitude over textured pixels (px)           lower is better
  dense_disp_p90   90th percentile of the same                                        lower is better
  sparse_disp_px   median displacement of RANSAC-verified SIFT matches (px)           lower is better
  match_rate       verified matches / min(#keypoints) (can fall under heavy weather)  diagnostic
  rel_rot_deg      essential-matrix relative rotation between the two views (deg)     lower is better
  edge_precision   fraction of edited-image edges lying within r px of an original edge
                   (hallucinated structure lowers it)                                 higher is better
  edge_recall      fraction of original edges still present (fog/night legitimately lower it)  diagnostic
  depth_spearman   rank correlation of monocular depth predictions (optional model)   higher is better
"""
from __future__ import annotations

import cv2
import numpy as np

from dgeo.match import fundamental_inliers, relative_pose, sift_matches, to_gray_u8


def _norm_gray(img: np.ndarray) -> np.ndarray:
    g = to_gray_u8(img)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)  # compensates exposure/fog contrast loss


def dense_drift(img0: np.ndarray, img1: np.ndarray, grad_pct: float = 70.0) -> dict:
    g0, g1 = _norm_gray(img0), _norm_gray(img1)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(g0, g1, None)
    mag = np.linalg.norm(flow, axis=-1)
    gx, gy = cv2.Sobel(g0, cv2.CV_32F, 1, 0), cv2.Sobel(g0, cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)
    sel = grad >= np.percentile(grad, grad_pct)
    m = mag[sel]
    return dict(dense_disp_px=float(np.median(m)), dense_disp_p90=float(np.percentile(m, 90)))


def sparse_drift(img0: np.ndarray, img1: np.ndarray) -> dict:
    p0, p1 = sift_matches(_norm_gray(img0), _norm_gray(img1))
    inl = fundamental_inliers(p0, p1)
    n_kp = max(len(p0), 1)
    out = dict(n_matches=int(len(p0)), n_inliers=int(inl.sum()), match_rate=float(inl.sum() / n_kp))
    if inl.sum() >= 8:
        d = np.linalg.norm(p1[inl] - p0[inl], axis=1)
        out["sparse_disp_px"] = float(np.median(d))
    else:
        out["sparse_disp_px"] = float("nan")
    return out


def pose_drift(img0: np.ndarray, img1: np.ndarray, focal_frac: float = 0.8) -> dict:
    h, w = img0.shape[:2]
    f = focal_frac * w
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    p0, p1 = sift_matches(_norm_gray(img0), _norm_gray(img1))
    R, _, n = relative_pose(p0, p1, K)
    if R is None:
        return dict(rel_rot_deg=float("nan"), pose_inliers=0)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return dict(rel_rot_deg=ang, pose_inliers=int(n))


def edge_consistency(img0: np.ndarray, img1: np.ndarray, radius: int = 2) -> dict:
    g0, g1 = _norm_gray(img0), _norm_gray(img1)
    e0 = cv2.Canny(cv2.GaussianBlur(g0, (3, 3), 0), 60, 140) > 0
    e1 = cv2.Canny(cv2.GaussianBlur(g1, (3, 3), 0), 60, 140) > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    d0 = cv2.dilate(e0.astype(np.uint8), k) > 0
    d1 = cv2.dilate(e1.astype(np.uint8), k) > 0
    prec = float((e1 & d0).sum() / max(e1.sum(), 1))
    rec = float((e0 & d1).sum() / max(e0.sum(), 1))
    return dict(edge_precision=prec, edge_recall=rec)


_DEPTH_PIPE = None


def depth_spearman(img0: np.ndarray, img1: np.ndarray, model: str = "depth-anything/Depth-Anything-V2-Small-hf") -> dict:
    """Optional: needs `transformers` + a downloadable depth model (CPU works, slow)."""
    global _DEPTH_PIPE
    from PIL import Image
    from scipy.stats import spearmanr
    from transformers import pipeline

    if _DEPTH_PIPE is None:
        _DEPTH_PIPE = pipeline("depth-estimation", model=model)
    d0 = np.array(_DEPTH_PIPE(Image.fromarray(img0))["depth"], dtype=np.float32)
    d1 = np.array(_DEPTH_PIPE(Image.fromarray(img1))["depth"], dtype=np.float32)
    d1 = cv2.resize(d1, (d0.shape[1], d0.shape[0]))
    return dict(depth_spearman=float(spearmanr(d0.ravel(), d1.ravel()).statistic))


def all_metrics(img0: np.ndarray, img1: np.ndarray, with_depth: bool = False) -> dict:
    out = {}
    out.update(dense_drift(img0, img1))
    out.update(sparse_drift(img0, img1))
    out.update(pose_drift(img0, img1))
    out.update(edge_consistency(img0, img1))
    if with_depth:
        out.update(depth_spearman(img0, img1))
    return out
