"""Controlled image edits with a KNOWN effect on geometry, used to validate the drift metrics.

benign (should score as no drift):   fog, night, rain, blur
corrupting (should score as drift):  shift, rotate, elastic, hallucinate (adds structures)
All functions take/return uint8 RGB.
"""
from __future__ import annotations

import cv2
import numpy as np


def _ground_depth(h: int, w: int, horizon_frac: float = 0.45) -> np.ndarray:
    """Depth proxy from the ground-plane assumption: depth ~ 1 / (v - v_horizon), sky at max depth."""
    v = np.arange(h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    vh = horizon_frac * h
    depth = np.where(v > vh + 1, 1.0 / (v - vh + 1e-3), np.nan)
    far = np.nanmax(depth[~np.isnan(depth)]) if np.isfinite(depth).any() else 1.0
    depth = np.where(np.isnan(depth), far, depth)
    return depth / depth.max()


def fog(img: np.ndarray, beta: float = 2.5, airlight: float = 0.85) -> np.ndarray:
    h, w = img.shape[:2]
    t = np.exp(-beta * _ground_depth(h, w))[..., None]
    out = img.astype(np.float32) / 255 * t + airlight * (1 - t)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def night(img: np.ndarray, gain: float = 0.25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (img.astype(np.float32) / 255) ** 1.6 * gain
    x *= np.array([0.8, 0.9, 1.2], np.float32)
    x += rng.normal(0, 0.012, x.shape)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def rain(img: np.ndarray, density: float = 0.002, length: int = 14, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    n = int(density * h * w)
    for _ in range(n):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.line(layer, (x, y), (x - 3, y + length), 1.0, 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    out = 0.8 * img.astype(np.float32) + 0.2 * 255 * layer[..., None] + 0.05 * 255
    return np.clip(cv2.GaussianBlur(out, (3, 3), 0), 0, 255).astype(np.uint8)


def blur(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def shift(img: np.ndarray, px: float) -> np.ndarray:
    m = np.float32([[1, 0, px], [0, 1, 0]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def elastic(img: np.ndarray, amp_px: float, seed: int = 0, smooth: float = 25.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), smooth)
    dy = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), smooth)
    scale = amp_px / max(np.abs(dx).max(), np.abs(dy).max(), 1e-6)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + dx * scale, gy + dy * scale, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def hallucinate(img: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Paste n random textured rectangles (structure that is not in the original)."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(n):
        bw, bh = int(rng.integers(w // 12, w // 5)), int(rng.integers(h // 12, h // 5))
        x, y = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
        patch = rng.integers(40, 220, size=(bh, bw, 3)).astype(np.uint8)
        patch = cv2.GaussianBlur(patch, (0, 0), 1.0)
        out[y:y + bh, x:x + bw] = patch
    return out


BENIGN = {
    "fog": [dict(beta=1.5), dict(beta=3.0), dict(beta=5.0)],
    "night": [dict(gain=0.6), dict(gain=0.35), dict(gain=0.2)],
    "rain": [dict(density=0.001), dict(density=0.003), dict(density=0.006)],
    "blur": [dict(sigma=1.0), dict(sigma=2.0), dict(sigma=3.0)],
}
CORRUPTING = {
    "shift": ("px", [0.5, 1, 2, 4, 8]),
    "rotate": ("deg", [0.1, 0.25, 0.5, 1.0, 2.0]),
    "elastic": ("amp_px", [0.5, 1, 2, 4, 8]),
}
FUNCS = dict(fog=fog, night=night, rain=rain, blur=blur, shift=shift, rotate=rotate, elastic=elastic)
