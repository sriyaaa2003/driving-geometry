"""Data-curation samplers over embedding matrices (rows = L2-normalised image embeddings).

Every sampler has the signature  f(emb, k, seed=0, **kw) -> np.ndarray of k unique row indices.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def random_sampler(emb: np.ndarray, k: int, seed: int = 0, **_) -> np.ndarray:
    return np.random.default_rng(seed).choice(len(emb), size=k, replace=False)


def _closest_to_centers(emb, centers, labels):
    """Index of the point closest to each center (within its own cluster); -1 for empty clusters."""
    out = np.full(len(centers), -1, dtype=int)
    best = np.full(len(centers), np.inf)
    d = 1.0 - np.einsum("nd,nd->n", emb, centers[labels])
    for i, (lab, dist) in enumerate(zip(labels, d)):
        if dist < best[lab]:
            best[lab], out[lab] = dist, i
    return out


def kmeans_flat(emb: np.ndarray, k: int, seed: int = 0, **_) -> np.ndarray:
    """Flat k-means with k clusters; pick the member nearest each centroid (SemDeDup-style baseline)."""
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=1, batch_size=4096).fit(emb)
    idx = _closest_to_centers(emb, km.cluster_centers_, km.labels_)
    idx = np.unique(idx[idx >= 0])
    if len(idx) < k:  # empty clusters: top up randomly
        rest = np.setdiff1d(np.arange(len(emb)), idx)
        idx = np.concatenate([idx, np.random.default_rng(seed).choice(rest, k - len(idx), replace=False)])
    return idx[:k]


def kcenter_greedy(emb: np.ndarray, k: int, seed: int = 0, **_) -> np.ndarray:
    """Greedy k-center / farthest-point coreset under cosine distance."""
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(emb)))
    sel = [first]
    mind = 1.0 - emb @ emb[first]
    for _ in range(k - 1):
        i = int(np.argmax(mind))
        sel.append(i)
        mind = np.minimum(mind, 1.0 - emb @ emb[i])
    return np.array(sel)


def water_fill(budget: int, caps: np.ndarray, weights: np.ndarray, rng) -> np.ndarray:
    """Integer allocation of `budget` across children proportional to `weights`, never exceeding `caps`."""
    n = len(caps)
    alloc = np.zeros(n, dtype=int)
    remaining = int(min(budget, caps.sum()))
    while remaining > 0:
        active = alloc < caps
        w = weights * active
        share = remaining * w / w.sum()
        add = np.minimum(np.floor(share).astype(int), caps - alloc)
        if add.sum() == 0:  # hand out the remainder one by one, largest fractional share first
            frac = (share - np.floor(share)) + 1e-6 * rng.random(n)
            for i in np.argsort(-frac):
                if remaining == 0:
                    break
                if active[i]:
                    alloc[i] += 1
                    remaining -= 1
        else:
            alloc += add
            remaining -= int(add.sum())
    return alloc


def build_kmeans_tree(emb: np.ndarray, level_sizes, seed: int = 0):
    """Bottom-up hierarchical k-means: level 1 clusters the points, level l+1 clusters level-l centroids."""
    labels = []
    centers = []
    cur = emb
    for li, c in enumerate(level_sizes):
        c = int(min(c, len(cur)))
        if li == 0:
            km = MiniBatchKMeans(n_clusters=c, random_state=seed, n_init=1, batch_size=4096).fit(cur)
        else:
            km = KMeans(n_clusters=c, random_state=seed, n_init=1, max_iter=50).fit(cur)
        lab = km.labels_
        uniq, lab = np.unique(lab, return_inverse=True)  # drop empty clusters
        ctr = np.stack([cur[lab == j].mean(0) for j in range(len(uniq))])
        ctr /= np.linalg.norm(ctr, axis=1, keepdims=True) + 1e-12
        labels.append(lab)
        centers.append(ctr)
        cur = ctr
    return labels, centers


def default_levels(n: int, k: int, branching: int = 4, leaf_mult: int = 8, min_top: int = 6):
    """Leaf level much finer than the budget (leaf_mult * k, capped at n // 4), shrinking by `branching`
    per level until the top level would drop below `min_top` clusters. Balancing at the upper levels is what
    lets rare scenes survive; with leaves == k the scheme degenerates to flat k-means."""
    c = int(min(max(leaf_mult * k, 8), max(n // 4, min(k, n // 2))))
    levels = [c]
    while levels[-1] // branching >= min_top:
        levels.append(levels[-1] // branching)
    return levels


def hierarchical_kmeans(emb: np.ndarray, k: int, seed: int = 0, levels=None, alpha: float = 0.0,
                        **_) -> np.ndarray:
    """Hierarchical k-means sampling. The budget is split top-down; at each node children receive
    budget proportional to (points beneath)^alpha. alpha=0: balanced across clusters at every level
    (favours rare scenes); alpha=1: proportional to size (statistically close to random).
    Inside a leaf cluster the point nearest the centroid is taken first, the rest at random."""
    rng = np.random.default_rng(seed)
    levels = levels or default_levels(len(emb), k)
    labels, centers = build_kmeans_tree(emb, levels, seed)
    n_levels = len(labels)
    # number of points under every node, per level
    counts = [np.bincount(labels[0], minlength=len(centers[0]))]
    for li in range(1, n_levels):
        counts.append(np.bincount(labels[li], weights=counts[li - 1], minlength=len(centers[li])).astype(int))
    leaf_members = [np.where(labels[0] == j)[0] for j in range(len(centers[0]))]
    children = [None] + [
        [np.where(labels[li] == j)[0] for j in range(len(centers[li]))] for li in range(1, n_levels)
    ]  # children[li][j] -> node ids at level li-1

    selected = []

    def descend(li: int, node: int, budget: int):
        if budget <= 0:
            return
        if li == 0:
            members = leaf_members[node]
            budget = min(budget, len(members))
            d = 1.0 - emb[members] @ centers[0][node]
            order = np.argsort(d)
            first = members[order[0]]
            rest = members[order[1:]]
            pick = [first] + ([] if budget == 1 else list(rng.choice(rest, budget - 1, replace=False)))
            selected.extend(pick)
            return
        kids = children[li][node]
        caps = counts[li - 1][kids]
        alloc = water_fill(budget, caps, caps.astype(float) ** alpha, rng)
        for kid, b in zip(kids, alloc):
            descend(li - 1, int(kid), int(b))

    top = n_levels - 1
    top_caps = counts[top]
    top_alloc = water_fill(k, top_caps, top_caps.astype(float) ** alpha, rng)
    for node, b in enumerate(top_alloc):
        descend(top, node, int(b))
    sel = np.array(selected, dtype=int)
    assert len(np.unique(sel)) == len(sel)
    return sel


SAMPLERS = {
    "random": random_sampler,
    "kmeans_flat": kmeans_flat,
    "kcenter": kcenter_greedy,
    "hier_balanced": lambda e, k, seed=0, **kw: hierarchical_kmeans(e, k, seed, alpha=0.0, **kw),
    "hier_sqrt": lambda e, k, seed=0, **kw: hierarchical_kmeans(e, k, seed, alpha=0.5, **kw),
}

from dgeo.curation.baselines import d4, proto_hard, semdedup  # noqa: E402

SAMPLERS.update({"semdedup": semdedup, "d4": d4, "proto_hard": proto_hard})
