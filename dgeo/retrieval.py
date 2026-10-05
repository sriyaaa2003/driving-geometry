from __future__ import annotations

import numpy as np


def knn_cosine(queries: np.ndarray, db: np.ndarray, k: int = 10, chunk: int = 2048):
    """Top-k cosine neighbours (inputs assumed L2-normalised). Returns indices (nq,k) and sims (nq,k)."""
    k = min(k, db.shape[0])
    idx_all, sim_all = [], []
    for i in range(0, len(queries), chunk):
        s = queries[i:i + chunk] @ db.T
        if k < db.shape[0]:
            part = np.argpartition(-s, k - 1, axis=1)[:, :k]
        else:
            part = np.tile(np.arange(k), (s.shape[0], 1))
        ps = np.take_along_axis(s, part, 1)
        order = np.argsort(-ps, axis=1)
        idx_all.append(np.take_along_axis(part, order, 1))
        sim_all.append(np.take_along_axis(ps, order, 1))
    return np.concatenate(idx_all), np.concatenate(sim_all)


def recall_at_k(q_xy: np.ndarray, db_xy: np.ndarray, retrieved: np.ndarray, thr_m: float = 25.0,
                ks=(1, 5, 10)) -> dict:
    """Place-recognition recall@k: a query is correct if any of its top-k is within thr_m metres."""
    d = np.linalg.norm(db_xy[retrieved] - q_xy[:, None, :], axis=-1)  # (nq,k)
    out = {}
    for k in ks:
        kk = min(k, retrieved.shape[1])
        out[f"R@{k}"] = float((d[:, :kk] <= thr_m).any(axis=1).mean())
    return out


def per_query_hit(q_xy, db_xy, retrieved, thr_m=25.0, k=1) -> np.ndarray:
    d = np.linalg.norm(db_xy[retrieved[:, :k]] - q_xy[:, None, :], axis=-1)
    return (d <= thr_m).any(axis=1)
