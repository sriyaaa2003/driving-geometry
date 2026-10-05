"""Score original/edited image pairs (same file names in two folders) with the geometry-drift metrics.

  python -m dgeo.drift.score_pairs --orig data/imgs --edited data/edited/rainy_night --out results/drift/rainy_night
Add --with-depth to include the monocular-depth rank correlation (downloads a small depth model).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dgeo.drift.metrics import all_metrics
from dgeo.stats import bootstrap_ci, fmt_ci

SUMMARY = ["dense_disp_px", "dense_disp_p90", "sparse_disp_px", "match_rate", "rel_rot_deg", "edge_precision",
           "edge_recall", "depth_spearman"]


def read_rgb(p: Path, width: int = 480) -> np.ndarray:
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    return cv2.resize(im, (width, int(h * width / w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--edited", required=True)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--with-depth", action="store_true")
    ap.add_argument("--out", default="results/drift/pairs")
    a = ap.parse_args()
    exts = {".jpg", ".jpeg", ".png"}
    orig = {p.stem: p for p in Path(a.orig).iterdir() if p.suffix.lower() in exts}
    edit = {p.stem: p for p in Path(a.edited).iterdir() if p.suffix.lower() in exts}
    names = sorted(set(orig) & set(edit))
    if not names:
        raise SystemExit("no matching file names between the two folders")
    rows = []
    for n in names:
        i0 = read_rgb(orig[n], a.width)
        i1 = cv2.resize(read_rgb(edit[n], a.width), (i0.shape[1], i0.shape[0]))
        rows.append(dict(name=n, **all_metrics(i0, i1, a.with_depth)))
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out) + ".csv", index=False)
    lines = [f"# Drift summary ({len(df)} pairs): {a.edited}", "", "| metric | mean [95% bootstrap CI] |", "|---|---|"]
    for m in SUMMARY:
        if m in df:
            lines.append(f"| {m} | {fmt_ci(*bootstrap_ci(df[m].values), 3)} |")
    Path(str(out) + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
