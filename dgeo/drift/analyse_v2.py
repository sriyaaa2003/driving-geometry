"""Analysis of results/drift_v2/scores.csv: is the drift metric trustworthy, and what did the generators do?

    python -m dgeo.drift.analyse_v2 --scores results/drift_v2/scores.csv --out results/drift_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dgeo.stats import bootstrap_ci, fmt_ci

JUDGES = {  # name -> (column, sign): sign -1 means a LOW value signals moved geometry
    "drift: dense flow (px)": ("dense_disp_px", 1),
    "drift: sparse SIFT (px)": ("sparse_disp_px", 1),
    "drift: essential-matrix rotation (deg)": ("rel_rot_deg", 1),
    "edge precision": ("edge_precision", -1),
    "segmentation pixel agreement": ("seg_pixel_agreement", -1),
    "segmentation IoU, road+building+sky+...": ("seg_iou_structure", -1),
    "depth rank correlation": ("depth_spearman", -1),
}
GEN_ORDER = ["noop_depth", "i2i", "depth", "canny", "depth_canny"]
GEN_NAME = {"noop_depth": "depth ControlNet, neutral prompt (generator floor)", "i2i": "img2img 0.5, no control",
            "depth": "depth ControlNet", "canny": "canny ControlNet", "depth_canny": "depth + canny ControlNet"}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def geometry_moved(d: pd.DataFrame) -> pd.Series:
    thr = {"shift": 2.0, "rotate": 0.5, "elastic": 2.0}
    return d.apply(lambda r: r.source == "corrupt" and r.magnitude >= thr[r.kind], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/drift_v2/scores.csv")
    ap.add_argument("--out", default="results/drift_v2")
    a = ap.parse_args()
    d = pd.read_csv(a.scores)
    out = Path(a.out)
    n_img = d.image.nunique()
    L = ["# Drift metric v2: checked against independent judges", "",
         f"{n_img} real BDD100K photos. All pairs are scored at the same width (480 px), generated or synthetic. "
         "Judges that do not use optical flow: semantic-segmentation agreement (SegFormer, Cityscapes) and "
         "monocular-depth rank correlation (Depth Anything V2).", ""]

    # 1. can the metrics tell moved geometry from appearance-only edits? (known ground truth)
    syn = d[d.source.isin(["benign", "corrupt"])].copy()
    syn["moved"] = geometry_moved(syn)
    syn = syn[(syn.source == "benign") | syn.moved]
    L += ["## 1. Detecting geometry that really moved (synthetic edits with known effect)", "",
          "Positives: shifts of at least 2 px, rotations of at least 0.5 deg, elastic warps of at least 2 px. "
          "Negatives: fog, night, rain and blur of three strengths. AUC 1.0 = perfect, 0.5 = chance. "
          "95% bootstrap CI over images.", "",
          "| measure | AUC |", "|---|---|"]
    imgs = syn.image.unique()
    rng = np.random.default_rng(0)
    for name, (col, sign) in JUDGES.items():
        def one(sub):
            p = sub[sub.moved][col].to_numpy() * sign
            n = sub[~sub.moved][col].to_numpy() * sign
            return auc(p, n)
        pt = one(syn)
        boots = [one(pd.concat([syn[syn.image == i] for i in rng.choice(imgs, len(imgs))])) for _ in range(200)]
        L.append(f"| {name} | {pt:.3f} [{np.nanpercentile(boots, 2.5):.3f}, {np.nanpercentile(boots, 97.5):.3f}] |")

    # 2. generators
    gen = d[d.source == "generated"].copy()
    L += ["", "## 2. What the generators do (medians over images, prompts and seeds)", "",
          "| generator | dense drift (px) | sparse drift (px) | edge precision | seg. pixel agreement | "
          "seg. structure IoU | depth rank corr. |", "|---|---|---|---|---|---|---|"]
    for v in GEN_ORDER:
        g = gen[gen.kind == v]
        if not len(g):
            continue
        L.append(f"| {GEN_NAME[v]} | " + " | ".join(
            f"{g[c].median():.2f}" for c in ["dense_disp_px", "sparse_disp_px", "edge_precision",
                                              "seg_pixel_agreement", "seg_iou_structure", "depth_spearman"]) + " |")
    floor = gen[gen.kind == "noop_depth"].groupby("image").dense_disp_px.median()
    dep = gen[gen.kind == "depth"].groupby(["image", "setting"]).dense_disp_px.median().reset_index()
    dep["floor"] = dep.image.map(floor)
    L += ["", "Weather prompt vs the generator floor (depth ControlNet, dense drift in px, mean over images):", "",
          "| prompt | with weather prompt | neutral prompt (floor) | difference [95% CI] |", "|---|---|---|---|"]
    for s, g in dep.groupby("setting"):
        diff = (g.dense_disp_px - g.floor).to_numpy()
        L.append(f"| {s} | {g.dense_disp_px.mean():.2f} | {g.floor.mean():.2f} | {fmt_ci(*bootstrap_ci(diff), 2)} |")

    # 3. does the drift metric agree with the judges on the generated images?
    L += ["", "## 3. On generated images, does drift agree with the independent judges?", "",
          "Spearman correlation across all generated edits between dense drift and each judge "
          "(a judge column is flipped so that higher = more change).", "",
          "| judge | rho with dense drift |", "|---|---|"]
    for name, (col, sign) in JUDGES.items():
        if col == "dense_disp_px":
            continue
        ok = gen[["dense_disp_px", col]].dropna()
        L.append(f"| {name} | {spearmanr(ok.dense_disp_px, ok[col] * sign).statistic:.2f} |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    data = [gen[gen.kind == v].dense_disp_px.dropna().to_numpy() for v in GEN_ORDER if (gen.kind == v).any()]
    ax[0].boxplot(data, labels=[v for v in GEN_ORDER if (gen.kind == v).any()], showfliers=False)
    ax[0].set_ylabel("dense drift (px, 480 px wide)")
    ax[0].set_title("Generators: noop floor vs weather edits")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].scatter(gen.dense_disp_px, 1 - gen.seg_pixel_agreement, s=6, alpha=0.4)
    ax[1].set_xlabel("dense drift (px)")
    ax[1].set_ylabel("1 - segmentation agreement")
    ax[1].set_title("Generated edits: drift vs segmentation judge")
    plt.tight_layout()
    plt.savefig(out / "drift_v2.png", dpi=130)


if __name__ == "__main__":
    main()
