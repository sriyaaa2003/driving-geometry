"""Curation v2: published baselines, rare scenes without 'undefined', and a downstream probe test.

  python -m dgeo.curation.run_v2 --emb dinov2=data/bdd/emb_dinov2-small.npz clip=data/bdd/emb_clip-vit-base-patch32.npz \
      --manifest data/bdd/manifest.csv --out results/curation_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dgeo.curation import downstream
from dgeo.curation.run import load_embeddings, model_agreement, run_benchmark, summarise, to_markdown
from dgeo.stats import bootstrap_ci, fmt_ci

SAMPLERS_V2 = ["random", "kmeans_flat", "semdedup", "d4", "proto_hard", "hier_sqrt", "hier_balanced", "kcenter"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="results/curation_v2")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--coverage-budgets", default="0.01,0.05")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()
    man = pd.read_csv(a.manifest)
    embs = load_embeddings(a.emb, man)
    man["condition"] = man.weather.astype(str) + " | " + man.time_of_day.astype(str)
    man.loc[man.weather.isin(["undefined"]) | man.time_of_day.isin(["undefined"]), "condition"] = "undefined"
    tags = {t: man[t].fillna("NA").astype(str).values for t in ("scene", "weather", "time_of_day", "condition")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. coverage of rare values on all images (as before, but 'undefined' is no longer a rare scene)
    cov = run_benchmark(embs, tags, [float(b) for b in a.coverage_budgets.split(",")], a.seeds, SAMPLERS_V2)
    cov.to_csv(out / "coverage_raw.csv", index=False)
    summ = summarise(cov)
    summ.to_csv(out / "coverage_summary.csv", index=False)
    (out / "coverage_report.md").write_text(to_markdown(summ, model_agreement(embs)), encoding="utf-8")

    # 2. downstream probe on a held-out split
    ds = downstream.run(embs, man, tags, a.budget, SAMPLERS_V2, a.splits, a.seeds)
    ds.to_csv(out / "downstream_raw.csv", index=False)
    lines = ["# Downstream test: probe trained on the selected 5% of the training pool", "",
             f"{a.splits} grouped 70/30 splits, up to {a.seeds} sampler seeds each. The probe uses the OTHER "
             "model's features than the sampler (cross-model) or the same ones (same-model). "
             "Mean [95% bootstrap CI].", ""]
    for same in (False, True):
        for tag in tags:
            g = ds[(ds.same_model == same) & (ds.tag == tag)]
            lines += [f"## {'same' if same else 'cross'}-model, tag={tag}", "",
                      "| sampler | macro-F1 | rare-class recall |", "|---|---|---|"]
            for s in ["all_data"] + SAMPLERS_V2:
                gs = g[g.sampler == s]
                if len(gs):
                    lines.append(f"| {s} | {fmt_ci(*bootstrap_ci(gs.macro_f1.values), 3)} | "
                                 f"{fmt_ci(*bootstrap_ci(gs.rare_recall.values), 3)} |")
            lines.append("")
    (out / "downstream_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
