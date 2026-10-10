"""Curation benchmark: samplers x budgets x seeds x models, scored on held-out scenario tags.

Real data:
  python -m dgeo.curation.run --emb dinov2=data/emb_dinov2.npz clip=data/emb_clip.npz \
      --manifest data/manifest.csv --tags weather,time_of_day,country_code --out results/curation
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dgeo.curation.metrics import linear_cka, rank_agreement, rarity_scores, tag_metrics, top_overlap
from dgeo.curation.samplers import SAMPLERS
from dgeo.stats import bootstrap_ci, fmt_ci


def load_embeddings(specs, manifest: pd.DataFrame | None):
    """specs: ['name=path.npz', ...] -> {name: emb aligned to manifest rows}."""
    out = {}
    for s in specs:
        name, path = s.split("=", 1)
        z = np.load(path, allow_pickle=True)
        emb, ids = z["emb"], z["ids"].astype(str)
        if manifest is not None:
            pos = pd.Series(np.arange(len(ids)), index=ids)
            missing = set(manifest["id"].astype(str)) - set(pos.index)
            if missing:
                raise SystemExit(f"{name}: {len(missing)} manifest ids missing from {path}")
            emb = emb[pos.loc[manifest["id"].astype(str)].values]
        out[name] = emb
    return out


def run_benchmark(embs: dict, tags: dict, budgets, seeds: int, samplers=None, rare_frac: float = 0.05):
    samplers = samplers or list(SAMPLERS)
    rows = []
    for (mname, emb), frac in itertools.product(embs.items(), budgets):
        k = max(2, int(round(frac * len(emb))))
        for sname in samplers:
            for seed in range(seeds):
                sel = SAMPLERS[sname](emb, k, seed=seed)
                for tname, tarr in tags.items():
                    m = tag_metrics(sel, tarr, rare_frac)
                    rows.append(dict(model=mname, sampler=sname, budget=frac, k=k, seed=seed, tag=tname, **m))
        print(f"[curation] {mname} budget={frac:g} done", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for keys, g in df.groupby(["model", "tag", "budget", "sampler"]):
        rec = dict(zip(["model", "tag", "budget", "sampler"], keys))
        for metric in ["rare_coverage", "rare_lift", "entropy_ratio"]:
            p, lo, hi = bootstrap_ci(g[metric].values)
            rec[metric], rec[metric + "_lo"], rec[metric + "_hi"] = p, lo, hi
        recs.append(rec)
    return pd.DataFrame(recs)


def to_markdown(summary: pd.DataFrame, agreement: dict | None) -> str:
    lines = ["# Curation benchmark", "",
             "Scored on held-out tags (never used to build embeddings or samplers). "
             "Values: mean over seeds [95% bootstrap CI].", ""]
    for (model, tag), g in summary.groupby(["model", "tag"]):
        lines += [f"## model={model}, tag={tag}", "",
                  "| budget | sampler | rare_coverage | rare_lift | entropy_ratio |", "|---|---|---|---|---|"]
        for _, r in g.sort_values(["budget", "sampler"]).iterrows():
            lines.append(
                f"| {r.budget:g} | {r.sampler} | "
                f"{fmt_ci(r.rare_coverage, r.rare_coverage_lo, r.rare_coverage_hi, 2)} | "
                f"{fmt_ci(r.rare_lift, r.rare_lift_lo, r.rare_lift_hi, 2)} | "
                f"{fmt_ci(r.entropy_ratio, r.entropy_ratio_lo, r.entropy_ratio_hi, 2)} |")
        lines.append("")
    if agreement:
        lines += ["## Do the models agree on what is rare?", "",
                  "| pair | linear CKA | Spearman (rarity) | top-5% overlap (Jaccard) |", "|---|---|---|---|"]
        for pair, v in agreement.items():
            lines.append(f"| {pair} | {v['cka']:.3f} | {v['spearman']:.3f} | {v['top5_jaccard']:.3f} |")
    return "\n".join(lines) + "\n"


def model_agreement(embs: dict) -> dict:
    names = list(embs)
    rar = {n: rarity_scores(embs[n]) for n in names}
    out = {}
    for a, b in itertools.combinations(names, 2):
        out[f"{a} vs {b}"] = dict(cka=linear_cka(embs[a], embs[b]), spearman=rank_agreement(rar[a], rar[b]),
                                  top5_jaccard=top_overlap(rar[a], rar[b], 0.05))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", help="name=path.npz (npz has emb, ids)")
    ap.add_argument("--manifest", help="CSV with an 'id' column plus tag columns")
    ap.add_argument("--tags", default="", help="comma-separated tag columns")
    ap.add_argument("--budgets", default="0.01,0.05,0.1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rare-frac", type=float, default=0.05)
    ap.add_argument("--samplers", default=",".join(SAMPLERS))
    ap.add_argument("--out", default="results/curation")
    a = ap.parse_args()

    if True:
        if not a.emb or not a.manifest:
            raise SystemExit("need --emb and --manifest")
        man = pd.read_csv(a.manifest)
        embs = load_embeddings(a.emb, man)
        tags = {t: man[t].fillna("NA").astype(str).values for t in a.tags.split(",") if t}
    df = run_benchmark(embs, tags, [float(b) for b in a.budgets.split(",")], a.seeds,
                       a.samplers.split(","), a.rare_frac)
    summary = summarise(df)
    agreement = model_agreement(embs) if len(embs) > 1 else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "raw.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "agreement.json").write_text(json.dumps(agreement, indent=2) if agreement else "null")
    (out / "report.md").write_text(to_markdown(summary, agreement), encoding="utf-8")
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
