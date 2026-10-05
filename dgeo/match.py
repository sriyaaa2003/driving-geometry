"""Classical local matching + geometric verification (CPU baseline; swap in LightGlue on GPU)."""
from __future__ import annotations

import cv2
import numpy as np


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    return img


def sift_matches(img0: np.ndarray, img1: np.ndarray, ratio: float = 0.8, nfeatures: int = 2000):
    """Ratio-test SIFT matches. Returns (pts0 (n,2), pts1 (n,2)) float32."""
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    k0, d0 = sift.detectAndCompute(to_gray_u8(img0), None)
    k1, d1 = sift.detectAndCompute(to_gray_u8(img1), None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    p0 = np.float32([k0[m.queryIdx].pt for m in good]).reshape(-1, 2)
    p1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return p0, p1


def fundamental_inliers(p0: np.ndarray, p1: np.ndarray, thresh: float = 1.5):
    """RANSAC fundamental-matrix verification. Returns boolean inlier mask."""
    if len(p0) < 8:
        return np.zeros(len(p0), bool)
    try:
        _, mask = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, thresh, 0.999)
    except cv2.error:  # degenerate configuration (e.g. identical point sets)
        mask = None
    if mask is None:
        # identical / purely translational matches are consistent by construction: accept the near-static ones
        d = np.linalg.norm(p1 - p0, axis=1)
        return d <= max(thresh, float(np.median(d)) + 3 * thresh)
    return mask.ravel().astype(bool)


def relative_pose(p0: np.ndarray, p1: np.ndarray, K: np.ndarray, thresh: float = 1.0):
    """Essential-matrix relative pose (R, t_dir, n_inliers) from point matches; (None, None, 0) on failure."""
    if len(p0) < 8:
        return None, None, 0
    try:
        E, mask = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, thresh)
        if E is None or E.shape != (3, 3):
            return None, None, 0
        n, R, t, _ = cv2.recoverPose(E, p0, p1, K, mask=mask)
    except cv2.error:
        return None, None, 0
    return R, t.ravel(), int(n)


def verify_pair(img0, img1) -> int:
    """Number of geometrically verified SIFT matches between two images (re-ranking score)."""
    p0, p1 = sift_matches(img0, img1)
    return int(fundamental_inliers(p0, p1).sum())
