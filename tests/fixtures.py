"""Synthetic data generators so every pipeline runs end to end without downloading anything."""
from __future__ import annotations

import numpy as np


def make_synthetic_embeddings(n: int = 6000, d: int = 64, n_major: int = 12, n_rare: int = 6,
                              rare_frac: float = 0.03, seed: int = 0):
    """Long-tailed 'driving scene' embeddings. Returns emb (L2-normed) and a tags dict.

    tags: 'scenario' (major + rare scene types) and 'weather' (clear/overcast common, snow/fog rare).
    Embedding clusters follow `scenario`; weather is only weakly reflected, as with real data.
    """
    rng = np.random.default_rng(seed)
    n_rare_pts = int(n * rare_frac)
    n_major_pts = n - n_rare_pts
    sizes_major = np.maximum((rng.dirichlet(np.ones(n_major) * 3.0) * n_major_pts).astype(int), 5)
    sizes_rare = np.maximum((rng.dirichlet(np.ones(n_rare)) * n_rare_pts).astype(int), 3)
    centers = rng.normal(size=(n_major + n_rare, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    sizes = np.concatenate([sizes_major, sizes_rare])
    emb, scen = [], []
    for c, (ctr, s) in enumerate(zip(centers, sizes)):
        emb.append(ctr + 0.35 * rng.normal(size=(s, d)))
        scen += [f"scene_{c:02d}"] * s
    emb = np.concatenate(emb)
    scen = np.array(scen)
    is_rare_scene = np.array([int(s.split("_")[1]) >= n_major for s in scen])
    weather = np.where(rng.random(len(scen)) < 0.5, "clear", "overcast").astype(object)
    rare_w = rng.random(len(scen)) < np.where(is_rare_scene, 0.6, 0.01)
    weather[rare_w] = rng.choice(["snow", "fog"], rare_w.sum())
    emb = emb + 0.15 * (weather[:, None] == "snow") * rng.normal(size=(1, d))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    perm = rng.permutation(len(emb))
    return emb[perm].astype(np.float32), {"scenario": scen[perm], "weather": weather[perm].astype(str)}


def make_synthetic_vpr(n_db: int = 1500, n_q: int = 400, d: int = 64, area: float = 1500.0,
                       shift: float = 0.6, seed: int = 0):
    """Positions on a 2D area; embedding = smooth function of position. Queries get a domain shift
    (a fixed random direction added), mimicking cross-dataset appearance change."""
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(2, d))
    freqs = rng.uniform(0.002, 0.01, size=(d,))

    def feat(xy):
        return np.sin(xy @ proj * freqs) + 0.5 * np.cos(xy[:, :1] * freqs * 3)

    db_xy = rng.uniform(0, area, size=(n_db, 2))
    q_xy = rng.uniform(0, area, size=(n_q, 2))
    db = feat(db_xy) + 0.05 * rng.normal(size=(n_db, d))
    dom = rng.normal(size=(1, d))
    q = feat(q_xy) + shift * dom + 0.15 * rng.normal(size=(n_q, d))
    db = db / np.linalg.norm(db, axis=1, keepdims=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return db.astype(np.float32), db_xy, q.astype(np.float32), q_xy


def make_street_image(h: int = 240, w: int = 360, seed: int = 0) -> np.ndarray:
    """Textured pseudo-street image (sky / buildings / road) as uint8 RGB, for validating image metrics."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    horizon = int(h * 0.45)
    img[:horizon] = np.array([0.6, 0.75, 0.95])
    yy = np.linspace(0, 1, h - horizon)[:, None, None]
    img[horizon:] = 0.35 + 0.15 * yy
    x = 0
    while x < w:  # buildings with windows
        bw = int(rng.integers(25, 60))
        bh = int(rng.integers(int(h * 0.15), int(h * 0.42)))
        col = rng.uniform(0.25, 0.8, size=3)
        img[horizon - bh:horizon, x:x + bw] = col
        for wy in range(horizon - bh + 6, horizon - 6, 14):
            for wx in range(x + 5, min(x + bw - 8, w), 12):
                img[wy:wy + 7, wx:wx + 6] = rng.uniform(0.05, 0.95, size=3)
        x += bw
    for _ in range(8):  # lane markings converging to the horizon
        xb = int(rng.integers(0, w))
        for t in np.linspace(0, 1, 40):
            yy_ = int(horizon + t * (h - horizon - 1))
            xx_ = int(w / 2 + (xb - w / 2) * t)
            if 0 <= xx_ < w:
                img[yy_, max(0, xx_ - 1):xx_ + 2] = 0.95
    img += rng.normal(0, 0.01, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
