"""Tables and figure from results/calibration_av2/trials.csv.

    python -m dgeo.calibration.report --trials results/calibration_av2/trials.csv --out results/calibration_av2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dgeo.stats import bootstrap_ci, fmt_ci

AX = [("rot_x_deg", "roll about x (deg)"), ("rot_y_deg", "pitch about y (deg)"), ("rot_z_deg", "yaw about z (deg)"),
      ("trans_x_cm", "shift x, forward (cm)"), ("trans_y_cm", "shift y, left (cm)"), ("trans_z_cm", "shift z, up (cm)")]
CURVED_DEG = 8.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="results/calibration_av2/trials.csv")
    ap.add_argument("--out", default="results/calibration_av2")
    a = ap.parse_args()
    d = pd.read_csv(a.trials)
    d["curved"] = d.heading_change_deg >= CURVED_DEG
    out = Path(a.out)

    # one number per segment and camera: the median over the random starts
    seg = d.groupby(["log_id", "segment", "cam_b"]).median(numeric_only=True).reset_index()
    seg["curved"] = seg.heading_change_deg >= CURVED_DEG
    for k, _ in AX:
        seg[f"abserr_{k}"] = d.assign(e=d[f"err_{k}"].abs()).groupby(["log_id", "segment", "cam_b"]).e.median().values
    d["converged"] = (d[[f"err_rot_{x}_deg" for x in "xyz"]].abs().max(axis=1) < 0.1)

    lines = ["# Calibrating a camera from a real drive: which directions are constrained",
             "",
             f"Argoverse 2 (val), {d.log_id.nunique()} logs, {len(seg)} drive segments of 10 LiDAR sweeps (about 3 s), "
             f"{len(d)} solves. Camera A = ring_front_center (trusted), camera B = stereo_front_left or "
             "stereo_front_right, B's extrinsic is perturbed (0.5 deg and 5 cm per axis, random) and recovered "
             "from LiDAR-projected photometric consistency between the two views.",
             "",
             f"Segments with a heading change of at least {CURVED_DEG:.0f} degrees count as curved: "
             f"{int(seg.curved.sum())} curved, {int((~seg.curved).sum())} straight.",
             "",
             f"Solves that ended with all three rotations within 0.1 deg of the truth: {d.converged.mean() * 100:.0f}%.",
             "",
             "## Error left after the solve (median over starts, then mean over segments with 95% bootstrap CI)",
             "",
             "| axis | initial error (mean abs) | after solve, all | after solve, straight | after solve, curved | "
             "Cramer-Rao std (median) |",
             "|---|---|---|---|---|---|"]
    for k, name in AX:
        row = [name, f"{d[f'init_{k}'].abs().mean():.2f}"]
        for sel in (np.ones(len(seg), bool), ~seg.curved.values, seg.curved.values):
            v = seg.loc[sel, f"abserr_{k}"].to_numpy()
            row.append(fmt_ci(*bootstrap_ci(v), digits=2) if len(v) else "n/a")
        row.append(f"{d[f'crb_{k}'].median():.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Rotations are well determined; the translation along the direction of travel (x) is the weak one. "
              "The Cramer-Rao standard deviations (last column) are much smaller than the errors actually left, "
              "so the Jacobian-based uncertainty is optimistic here.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    seg.to_csv(out / "segments.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (kind, idx) in zip(axs, (("rotation", range(3)), ("translation", range(3, 6))), strict=True):
        w = 0.38
        for j, (lab, sel) in enumerate((("straight", ~seg.curved.values), ("curved", seg.curved.values))):
            vals = [seg.loc[sel, f"abserr_{AX[i][0]}"].median() if sel.any() else np.nan for i in idx]
            ax.bar(np.arange(3) + (j - 0.5) * w, vals, w, label=f"{lab} (n={int(sel.sum())})")
        ax.set_xticks(range(3))
        ax.set_xticklabels([AX[i][1] for i in idx], rotation=15)
        ax.set_ylabel("median abs error after solve (deg)" if kind == "rotation" else "median abs error after solve (cm)")
        ax.set_title(kind)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out / "error_by_axis.png", dpi=130)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
