"""Published data-pruning baselines on embeddings: SemDeDup, D4 and SSL Prototypes.

Same signature as the samplers in samplers.py: f(emb, k, seed=0, **kw) -> k unique row indices.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def _kmeans(emb: np.ndarray, c: int, seed: int):
    km = MiniBatchKMeans(n_clusters=c, random_state=seed, n_init=1, batch_size=4096).fit(emb)
    ctr = km.cluster_centers_ / (np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True) + 1e-12)
    return km.labels_, ctr


def _cluster_sims(emb, labels, centers):
    """Per cluster: member indices ordered from the farthest to the nearest to the centroid, and the
    pairwise cosine matrix in that order (computed once, reused for every threshold)."""
    out = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        order = idx[np.argsort(emb[idx] @ centers[c])]
        out.append((order, emb[order] @ emb[order].T))
    return out


def _dedup(clusters, tau):
    """Inside each cluster keep a point only if its cosine similarity to every kept point is below tau."""
    keep = []
    for order, S in clusters:
        gone = np.zeros(len(order), dtype=bool)
        close = S >= tau
        for i in range(len(order)):
            if not gone[i]:
                keep.append(order[i])
                gone |= close[i]
    return np.array(keep, dtype=int)


def _fit_to_size(sel: np.ndarray, n: int, k: int, rng) -> np.ndarray:
    if len(sel) > k:
        return rng.choice(sel, k, replace=False)
    if len(sel) < k:
        rest = np.setdiff1d(np.arange(n), sel)
        return np.concatenate([sel, rng.choice(rest, k - len(sel), replace=False)])
    return sel


def semdedup(emb: np.ndarray, k: int, seed: int = 0, n_clusters: int = 100, **_) -> np.ndarray:
    """SemDeDup (Abbas et al. 2023). The similarity threshold is searched so that about k points survive."""
    rng = np.random.default_rng(seed)
    labels, centers = _kmeans(emb, max(2, min(n_clusters, len(emb) // 5)), seed)
    clusters = _cluster_sims(emb, labels, centers)
    lo, hi, best = 0.5, 0.999, None
    for _ in range(12):
        tau = (lo + hi) / 2
        kept = _dedup(clusters, tau)
        if len(kept) >= k:
            best, hi = kept, tau  # smallest threshold so far that still keeps at least k points
        else:
            lo = tau
    if best is None:
        best = _dedup(clusters, hi)
    return _fit_to_size(best, len(emb), k, rng)


def proto_hard(emb: np.ndarray, k: int, seed: int = 0, n_clusters: int = 100, **_) -> np.ndarray:
    """SSL Prototypes (Sorscher et al. 2022): difficulty = distance to the nearest k-means prototype;
    keep the k hardest points."""
    labels, centers = _kmeans(emb, max(2, min(n_clusters, len(emb) // 5)), seed)
    dist = 1.0 - np.einsum("nd,nd->n", emb, centers[labels])
    return np.argsort(-dist)[:k]


def d4(emb: np.ndarray, k: int, seed: int = 0, n_clusters: int = 100, keep_far: float = 0.5, **_) -> np.ndarray:
    """D4 (Tirumala et al. 2023): SemDeDup, then re-cluster what is left and keep the points farthest from
    their centroid (keep_far = fraction of each cluster to retain)."""
    rng = np.random.default_rng(seed)
    dedup = semdedup(emb, min(len(emb), int(k / keep_far)), seed, n_clusters)
    sub = emb[dedup]
    labels, centers = _kmeans(sub, max(2, min(n_clusters, len(sub) // 5)), seed + 1)
    dist = 1.0 - np.einsum("nd,nd->n", sub, centers[labels])
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        far = idx[np.argsort(-dist[idx])][: max(1, int(np.ceil(keep_far * len(idx))))]
        keep.extend(far)
    return _fit_to_size(dedup[np.array(keep, dtype=int)], len(emb), k, rng)
