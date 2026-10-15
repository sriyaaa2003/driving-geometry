"""Study 5: data curation on the Zenseact Open Dataset (ZOD), 100,000 real frames.

Inputs: DINOv2-small and CLIP ViT-B/32 embeddings of every frame (scripts/modal_zod_curation.py::embed) and the frame index
built from ZOD's own metadata (country, time, day/night, road condition, road type).

    python -m dgeo.zod.run_curation --emb dinov2=emb_dinov2-small.npz clip=emb_clip-vit-base-patch32.npz \
        --index index.parquet --out results/zod_curation

Samplers run on a 128-dimension PCA of each embedding (fitted on all 100,000 frames, no labels) so that the hierarchical
k-means sampler stays tractable at 100,000 frames; the hierarchical samplers use a leaf level of 4x the budget.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

from dgeo.curation.downstream import probe_scores
from dgeo.curation.metrics import rare_values, tag_metrics
from dgeo.curation.run import model_agreement, summarise, to_markdown
from dgeo.curation.samplers import SAMPLERS
from dgeo.stats import bootstrap_ci, fmt_ci

SAMPLER_NAMES = ["random", "kmeans_flat", "semdedup", "d4", "proto_hard", "hier_sqrt", "hier_balanced", "kcenter"]
HIER = ("hier_sqrt", "hier_balanced")


def select(names, emb, k, seed, leaf_mult):
    """Run one or two samplers on the same data; the hierarchical pair is fitted once (shared k-means tree)."""
    return {n: SAMPLERS[n](emb, k, seed=seed, leaf_mult=leaf_mult) for n in names}


def tasks_for(seeds):
    """(sampler group, seed) pairs: the two hierarchical samplers travel together; SSL Prototypes is deterministic."""
    out = []
    for n in SAMPLER_NAMES:
        if n in HIER:
            continue
        for s in range(1 if n == "proto_hard" else seeds):
            out.append(((n,), s))
    out += [(HIER, s) for s in range(seeds)]
    return out


def probe(pm, embs, sel, te, tags, rare, tname):
    return probe_scores(embs[pm][sel], tags[tname][sel], embs[pm][te], tags[tname][te], rare[tname])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", default="results/zod_curation")
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--leaf-mult", type=int, default=4)
    ap.add_argument("--coverage-budgets", default="0.01,0.05")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = {s.split("=", 1)[0]: np.load(s.split("=", 1)[1])["emb"] for s in a.emb}
    names = list(raw)
    idx = pd.read_parquet(a.index).sort_values("row").reset_index(drop=True)
    assert len(idx) == len(raw[names[0]])
    embs = {}
    for n, e in raw.items():
        x = PCA(a.pca, random_state=0, svd_solver="randomized").fit_transform(e)
        embs[n] = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    tags = {"condition": idx["cond"].astype(str).values, "road_type": idx["label_road_type"].astype(str).values,
            "country": idx["country"].astype(str).values}
    # frames of one drive are close in time: group by country and day so a drive never straddles a split
    groups = (idx["country"].astype(str) + "_" + (idx["ts"] // 86400).astype(int).astype(str)).values
    print(f"{len(idx)} frames, PCA {a.pca}, tags {({t: len(set(v)) for t, v in tags.items()})} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 1. coverage of rare values over all 100,000 frames ---------------------------------------------------
    rows = []
    for frac in [float(b) for b in a.coverage_budgets.split(",")]:
        k = int(round(frac * len(idx)))
        jobs = [(m, g, s) for m in names for g, s in tasks_for(a.seeds)]
        res = Parallel(n_jobs=a.jobs)(delayed(select)(g, embs[m], k, s, a.leaf_mult) for m, g, s in jobs)
        for (m, g, s), r in zip(jobs, res):
            for sname, sel in r.items():
                for tname, tarr in tags.items():
                    rows.append(dict(model=m, sampler=sname, budget=frac, k=k, seed=s, tag=tname, **tag_metrics(sel, tarr, 0.05)))
        print(f"[coverage] budget {frac:g} done ({time.time() - t0:.0f}s)", flush=True)
    cov = pd.DataFrame(rows)
    cov.to_csv(out / "coverage_raw.csv", index=False)
    summ = summarise(cov)
    summ.to_csv(out / "coverage_summary.csv", index=False)
    agree = model_agreement(raw)
    (out / "agreement.json").write_text(json.dumps(agree, indent=2))
    (out / "coverage_report.md").write_text(to_markdown(summ, agree), encoding="utf-8")
    print(f"[coverage] report written ({time.time() - t0:.0f}s)", flush=True)

    # ---- 2. downstream: train a probe on the selected 5% of the training pool, score on held-out frames --------
    rows = []
    for split in range(a.splits):
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=split).split(np.zeros(len(groups)), groups=groups))
        k = int(round(a.budget * len(tr)))
        rare = {t: set(rare_values(v[tr], 0.05)) for t, v in tags.items()}
        # reference: probe trained on the whole training pool
        ref = Parallel(n_jobs=a.jobs)(delayed(probe)(pm, embs, tr, te, tags, rare, t) for pm in names for t in tags)
        for (pm, t), sc in zip(itertools.product(names, tags), ref):
            rows.append(dict(split=split, seed=0, select_model="-", probe_model=pm, same_model=False, sampler="all_data", tag=t, k=len(tr), **sc))
        jobs = [(sm, g, s) for sm in names for g, s in tasks_for(a.seeds)]
        sels = Parallel(n_jobs=a.jobs)(delayed(select)(g, embs[sm][tr], k, s, a.leaf_mult) for sm, g, s in jobs)
        score_jobs = []
        for (sm, g, s), r in zip(jobs, sels):
            for sname, loc in r.items():
                sel = tr[loc]
                for pm in [m for m in names if m != sm] + [sm]:
                    for t in tags:
                        score_jobs.append((sm, sname, s, pm, t, sel))
        scores = Parallel(n_jobs=a.jobs)(delayed(probe)(pm, embs, sel, te, tags, rare, t) for sm, sname, s, pm, t, sel in score_jobs)
        for (sm, sname, s, pm, t, sel), sc in zip(score_jobs, scores):
            rows.append(dict(split=split, seed=s, select_model=sm, probe_model=pm, same_model=pm == sm, sampler=sname, tag=t, k=len(sel), **sc))
        pd.DataFrame(rows).to_csv(out / "downstream_raw.csv", index=False)
        print(f"[downstream] split {split} done ({time.time() - t0:.0f}s)", flush=True)

    ds = pd.DataFrame(rows)
    lines = [f"# ZOD downstream test: probe trained on the selected {a.budget:.0%} of the training pool", "",
             f"{a.splits} grouped 70/30 splits (group = country + day), {a.seeds} sampler seeds each. Cross-model: the probe uses the "
             "OTHER model's features; same-model: the same model's. Mean [95% bootstrap CI]. `all_data` trains on the whole pool.", ""]
    for same in (False, True):
        for tag in tags:
            g = ds[((ds.same_model == same) | (ds.sampler == "all_data")) & (ds.tag == tag)]
            lines += [f"## {'same' if same else 'cross'}-model, tag={tag}", "", "| sampler | macro-F1 | rare-class recall |", "|---|---|---|"]
            for s in ["all_data"] + SAMPLER_NAMES:
                gs = g[g.sampler == s]
                if len(gs):
                    lines.append(f"| {s} | {fmt_ci(*bootstrap_ci(gs.macro_f1.values), 3)} | {fmt_ci(*bootstrap_ci(gs.rare_recall.values), 3)} |")
            lines.append("")
    (out / "downstream_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"ALL DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
