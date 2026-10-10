"""Validate the geometry-drift metrics BEFORE trusting them on generated images.

Checks (all on images with known edits):
  1. dose-response: drift metrics rise monotonically with injected geometric corruption (Spearman rho per metric)
  2. specificity:   benign appearance edits (fog / night / rain / blur) score far below corruptions of >= 2 px
  3. hallucination: edge_precision falls when structure that is not in the original is pasted in

  python -m dgeo.drift.validate_metric --images data/bdd/drift_images   # real dashcam photos, edits applied on top, CPU
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

from dgeo.drift import synth_edits as E
from dgeo.drift.metrics import all_metrics
from dgeo.stats import bootstrap_ci, fmt_ci

DRIFT_METRICS = ["dense_disp_px", "dense_disp_p90", "sparse_disp_px", "rel_rot_deg"]


def load_images(folder: str | None, n_synth: int):
    imgs = []
    if folder:
        for p in sorted(Path(folder).glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                h, w = im.shape[:2]
                s = 360 / w
                imgs.append((p.stem, cv2.resize(im, (360, int(h * s)))))
    else:
        raise SystemExit("pass --images <folder of real photos>")
    return imgs


def run(images, with_depth: bool = False):
    rows = []
    for name, img in images:
        for kind, grid in E.BENIGN.items():
            for level, kw in enumerate(grid):
                m = all_metrics(img, E.FUNCS[kind](img, **kw), with_depth)
                rows.append(dict(image=name, family="benign", kind=kind, level=level, magnitude=level, **m))
        for kind, (arg, mags) in E.CORRUPTING.items():
            for mag in mags:
                edited = E.FUNCS[kind](img, **{arg: mag})
                m = all_metrics(img, edited, with_depth)
                rows.append(dict(image=name, family="corrupt", kind=kind, level=-1, magnitude=mag, **m))
        for n in (1, 3, 6):
            m = all_metrics(img, E.hallucinate(img, n), with_depth)
            rows.append(dict(image=name, family="halluc", kind="hallucinate", level=-1, magnitude=n, **m))
    return rows


def analyse(rows):
    res = {"dose_response": {}, "specificity": {}, "hallucination": {}}
    for kind in E.CORRUPTING:
        for met in DRIFT_METRICS:
            xs = [(r["magnitude"], r[met]) for r in rows if r["kind"] == kind and not np.isnan(r[met])]
            if len(xs) > 3:
                res["dose_response"][f"{kind}:{met}"] = float(spearmanr(*zip(*xs)).statistic)
    for met in ["dense_disp_px", "sparse_disp_px"]:
        ben = np.array([r[met] for r in rows if r["family"] == "benign" and not np.isnan(r[met])])
        cor = np.array([r[met] for r in rows if r["family"] == "corrupt" and r["magnitude"] >= 2 and not np.isnan(r[met])])
        res["specificity"][met] = dict(
            benign_median=float(np.median(ben)), benign_p90=float(np.percentile(ben, 90)),
            corrupt_ge2px_median=float(np.median(cor)), separation_ratio=float(np.median(cor) / max(np.median(ben), 1e-6)),
            benign_nan_frac=float(np.mean([np.isnan(r[met]) for r in rows if r["family"] == "benign"])))
    for n in (1, 3, 6):
        v = [r["edge_precision"] for r in rows if r["kind"] == "hallucinate" and r["magnitude"] == n]
        res["hallucination"][f"n={n}"] = bootstrap_ci(v)
    res["hallucination"]["benign_edge_precision"] = bootstrap_ci([r["edge_precision"] for r in rows if r["family"] == "benign"])
    return res


def to_markdown(res, rows) -> str:
    L = ["# Geometry-drift metric validation", "",
         "Images with KNOWN edits. A trustworthy metric must (1) rise with injected geometric corruption, "
         "(2) stay low for benign appearance edits, (3) notice hallucinated structure.", "",
         f"Images: {len({r['image'] for r in rows})}, edits per image: {len(rows) // len({r['image'] for r in rows})}.", "",
         "## 1. Dose-response (Spearman rho between injected magnitude and metric; 1.0 = perfect)", "",
         "| corruption : metric | rho |", "|---|---|"]
    for k, v in sorted(res["dose_response"].items()):
        L.append(f"| {k} | {v:.2f} |")
    L += ["", "## 2. Benign vs corrupting (px)", "",
          "| metric | benign median | benign p90 | corrupt (>=2 px) median | separation ratio | benign NaN frac |", "|---|---|---|---|---|---|"]
    for k, v in res["specificity"].items():
        L.append(f"| {k} | {v['benign_median']:.2f} | {v['benign_p90']:.2f} | {v['corrupt_ge2px_median']:.2f} | "
                 f"{v['separation_ratio']:.1f}x | {v['benign_nan_frac']:.2f} |")
    L += ["", "## 3. Hallucinated structure: edge precision (mean [95% CI]; higher = fewer hallucinations)", "",
          "| setting | edge_precision |", "|---|---|"]
    for k, v in res["hallucination"].items():
        L.append(f"| {k} | {fmt_ci(*v, 3)} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="folder of real images (optional)")
    ap.add_argument("--with-depth", action="store_true")
    ap.add_argument("--out", default="results/drift")
    a = ap.parse_args()
    rows = run(load_images(a.images, 0), a.with_depth)
    res = analyse(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(res, indent=2))
    (out / "validation.md").write_text(to_markdown(res, rows), encoding="utf-8")
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "validation_raw.csv", index=False)
    print(to_markdown(res, rows))


if __name__ == "__main__":
    main()
