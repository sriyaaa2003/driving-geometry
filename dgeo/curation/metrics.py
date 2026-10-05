"""Curation metrics scored on tags that were NOT used to build the embeddings or the samplers."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from dgeo.retrieval import knn_cosine


def rare_values(tag: np.ndarray, rare_frac: float = 0.05):
    vals, counts = np.unique(tag, return_counts=True)
    freq = counts / counts.sum()
    return {v: f for v, f in zip(vals, freq) if f <= rare_frac}


def tag_metrics(selected: np.ndarray, tag: np.ndarray, rare_frac: float = 0.05) -> dict:
    """For one tag column:
    rare_coverage  fraction of rare tag values (frequency <= rare_frac) with >= 1 selected sample
    rare_lift      mean over rare values of (share of selection) / (share of dataset); 1.0 = same as random
    entropy_ratio  Shannon entropy of the selection's tag distribution / entropy of the full dataset's
    """
    n, k = len(tag), len(selected)
    sel_tag = tag[selected]
    rare = rare_values(tag, rare_frac)
    out = {"rare_coverage": np.nan, "rare_lift": np.nan}
    if rare:
        cov, lift = [], []
        for v, f in rare.items():
            c = int((sel_tag == v).sum())
            cov.append(c >= 1)
            lift.append((c / k) / f)
        out["rare_coverage"] = float(np.mean(cov))
        out["rare_lift"] = float(np.mean(lift))

    def ent(a):
        _, c = np.unique(a, return_counts=True)
        p = c / c.sum()
        return float(-(p * np.log(p)).sum())

    h_full = ent(tag)
    out["entropy_ratio"] = ent(sel_tag) / h_full if h_full > 0 else np.nan
    return out


def rarity_scores(emb: np.ndarray, k: int = 10) -> np.ndarray:
    """Low-density score: 1 - mean cosine similarity to the k nearest other points. Higher = rarer."""
    idx, sims = knn_cosine(emb, emb, k=k + 1)
    return 1.0 - sims[:, 1:].mean(axis=1)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two representations of the same n samples."""
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    num = np.linalg.norm(y.T @ x, "fro") ** 2
    den = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(num / den)


def rank_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def top_overlap(a: np.ndarray, b: np.ndarray, frac: float = 0.05) -> float:
    """Jaccard overlap of the top-`frac` rarest samples according to two models."""
    m = max(1, int(len(a) * frac))
    ta, tb = set(np.argsort(-a)[:m]), set(np.argsort(-b)[:m])
    return len(ta & tb) / len(ta | tb)
