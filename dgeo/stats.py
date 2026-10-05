from __future__ import annotations

import numpy as np


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of `stat` over `values`. Returns (point, lo, hi)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    point = float(stat(v))
    if v.size == 1:
        return point, point, point
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.array([stat(v[i]) for i in idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def fmt_ci(point: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
