"""Does curation keep what localization needs? Build the reference database from a curated subset of
one dataset, query it with images from ANOTHER dataset, and measure place-recognition recall.

  python -m dgeo.curation.localization --db-emb data/emb_dinov2_zod.npz --db-manifest data/zod_manifest.csv \
      --q-emb data/emb_dinov2_msls.npz --q-manifest data/msls_manifest.csv --out results/localization
  python -m dgeo.curation.localization --demo
Manifests need columns: id, x, y (metres) -- or lat, lon (converted to local metres).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dgeo.curation.samplers import SAMPLERS
from dgeo.retrieval import knn_cosine, recall_at_k
from dgeo.stats import bootstrap_ci, fmt_ci


def latlon_to_xy(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    r = 6371000.0
    x = np.radians(lon - lon.mean()) * r * np.cos(np.radians(lat.mean()))
    y = np.radians(lat - lat.mean()) * r
    return np.stack([x, y], 1)


def xy_from_manifest(df: pd.DataFrame) -> np.ndarray:
    if {"x", "y"} <= set(df.columns):
        return df[["x", "y"]].to_numpy(float)
    if {"lat", "lon"} <= set(df.columns):
        return latlon_to_xy(df["lat"], df["lon"])
    raise SystemExit("manifest needs x,y or lat,lon columns")


def vpr_budget_sweep(db_emb, db_xy, q_emb, q_xy, budgets, samplers, seeds, thr_m=25.0, ks=(1, 5)):
    rows = []
    idx, _ = knn_cosine(q_emb, db_emb, k=max(ks))
    rows.append(dict(sampler="full_database", budget=1.0, seed=0, **recall_at_k(q_xy, db_xy, idx, thr_m, ks)))
    for frac in budgets:
        k = max(2, int(round(frac * len(db_emb))))
        for s in samplers:
            for seed in range(seeds):
                sel = SAMPLERS[s](db_emb, k, seed=seed)
                idx, _ = knn_cosine(q_emb, db_emb[sel], k=max(ks))
                rows.append(dict(sampler=s, budget=frac, seed=seed,
                                 **recall_at_k(q_xy, db_xy[sel], idx, thr_m, ks)))
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, thr_m: float) -> str:
    lines = ["# Localization recall under curation budgets", "",
             f"Reference DB = curated subset of the source dataset; queries from another dataset. "
             f"Correct if within {thr_m:g} m. Mean over seeds [95% bootstrap CI].", "",
             "| budget | sampler | R@1 | R@5 |", "|---|---|---|---|"]
    for (b, s), g in df.groupby(["budget", "sampler"]):
        r1 = bootstrap_ci(g["R@1"].values)
        r5 = bootstrap_ci(g["R@5"].values)
        lines.append(f"| {b:g} | {s} | {fmt_ci(*r1, 3)} | {fmt_ci(*r5, 3)} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-emb"), ap.add_argument("--db-manifest")
    ap.add_argument("--q-emb"), ap.add_argument("--q-manifest")
    ap.add_argument("--budgets", default="0.02,0.05,0.1,0.25")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--thr", type=float, default=25.0)
    ap.add_argument("--samplers", default=",".join(SAMPLERS))
    ap.add_argument("--out", default="results/localization")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        from dgeo.synth import make_synthetic_vpr

        db_emb, db_xy, q_emb, q_xy = make_synthetic_vpr()
    else:
        dbz, qz = np.load(a.db_emb, allow_pickle=True), np.load(a.q_emb, allow_pickle=True)
        dbm, qm = pd.read_csv(a.db_manifest), pd.read_csv(a.q_manifest)
        def align(z, m):
            pos = pd.Series(np.arange(len(z["ids"])), index=z["ids"].astype(str))
            return z["emb"][pos.loc[m["id"].astype(str)].values]
        db_emb, q_emb = align(dbz, dbm), align(qz, qm)
        db_xy, q_xy = xy_from_manifest(dbm), xy_from_manifest(qm)
    df = vpr_budget_sweep(db_emb, db_xy, q_emb, q_xy, [float(b) for b in a.budgets.split(",")],
                          a.samplers.split(","), a.seeds, a.thr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "raw.csv", index=False)
    (out / "report.md").write_text(report(df, a.thr), encoding="utf-8")
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
