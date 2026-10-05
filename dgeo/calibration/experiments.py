"""Calibration observability + convergence study on synthetic rigs (CPU, no data needed).

  python -m dgeo.calibration.experiments --quick     # ~2 min
  python -m dgeo.calibration.experiments             # ~15 min, writes results/calibration/

Questions:
 1. Which extrinsic directions does a drive constrain? (Gauss-Newton information at the true calibration)
 2. Does the path matter (straight vs curved)? Does scene structure matter (walls vs open road)?
 3. Does a road-plane mounting prior (camera height, pitch, roll) help?
 4. How large is the empirical error of a coarse-to-fine solve from a perturbed start?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dgeo.calibration.core import CalibProblem, RoadPrior, observability, perturbed_init, solve
from dgeo.calibration.scene import make_scene

SCENES = {
    "straight+walls": dict(path="straight", walls=True),
    "curve+walls": dict(path="curve", walls=True),
    "straight+open": dict(path="straight", walls=False),
    "curve+open": dict(path="curve", walls=False),
}
PRIOR = RoadPrior(sigma_h_m=0.02, sigma_angle_deg=0.3)
SIGMA_C = 0.05  # photometric residual scale (0.05 of the [0,1] range)


def study_observability():
    rows = []
    for name, kw in SCENES.items():
        scene = make_scene(**kw)
        for pname, pr in [("none", None), ("road-plane", PRIOR)]:
            o = observability(CalibProblem(scene, scene.R_cl, scene.t_cl, sigma_c=SIGMA_C, prior=pr))
            rows.append(dict(scene=name, prior=pname, lam_min=float(o["eigenvalues"][0]), cond=o["cond"],
                             common_mode_std_cm=o["common_mode_std_cm"],
                             rot_std_deg_max=float(o["crb_std"][:, :3].max()),
                             trans_std_cm_max=float(o["crb_std"][:, 3:].max()),
                             weakest=o["weakest_direction"][:2]))
            print(f"[obs] {name} / {pname}", flush=True)
    return rows


def study_convergence(perturbs, trials: int):
    rows = []
    for name, kw in SCENES.items():
        scene = make_scene(**kw)
        for pname, pr in [("none", None), ("road-plane", PRIOR)]:
            for rot, tr in perturbs:
                for trial in range(trials):
                    Rs, ts = perturbed_init(scene, rot, tr, seed=100 + trial)
                    pb = CalibProblem(scene, Rs, ts, sigma_c=SIGMA_C, prior=pr, prior_nominal=(scene.R_cl, scene.t_cl))
                    p = solve(pb)
                    err = pb.errors(p)
                    pos = np.abs(pb.position_error_lidar(p))  # (C,3) cm
                    rows.append(dict(scene=name, prior=pname, start_rot_deg=rot, start_trans_cm=tr, trial=trial,
                                     rot_err_deg=float(np.mean([e[0] for e in err])),
                                     trans_err_cm=float(np.mean([e[1] for e in err])),
                                     pos_err_lidar_cm={a: float(pos[:, i].mean()) for i, a in enumerate("xyz")}))
                    print(f"[conv] {name} / {pname} start=({rot} deg,{tr} cm) trial={trial}: "
                          f"rot {rows[-1]['rot_err_deg']:.2f} deg, trans {rows[-1]['trans_err_cm']:.1f} cm", flush=True)
    return rows


def report(obs, conv) -> str:
    L = ["# Calibration study (synthetic fisheye rig; CPU)", "",
         "Photometric multi-view consistency of LiDAR points with known odometry. **Synthetic data**: these are "
         "properties of the method on a controlled scene, not results on a real vehicle.", "",
         "## 1. Observability at the true calibration", "",
         "`common-mode std` = marginal std (cm) of shifting ALL cameras together along a LiDAR-frame axis. "
         "The linear (Cramer-Rao) numbers are optimistic: they ignore systematic model error, which is why the "
         "empirical errors in section 2 are larger.", "",
         "| scene | prior | lambda_min | common-mode std x / y / z (cm) | max rot std (deg) | max trans std (cm) |",
         "|---|---|---|---|---|---|"]
    for r in obs:
        c = r["common_mode_std_cm"]
        L.append(f"| {r['scene']} | {r['prior']} | {r['lam_min']:.2f} | {c['x']:.2f} / {c['y']:.2f} / {c['z']:.2f} | "
                 f"{r['rot_std_deg_max']:.3f} | {r['trans_std_cm_max']:.2f} |")
    L += ["", "## 2. Empirical error of a coarse-to-fine solve from a perturbed start", "",
          "Mean over cameras and trials. Position error is in the LiDAR frame.", "",
          "| scene | prior | start | rot err (deg) | trans err (cm) | pos err x / y / z (cm) |", "|---|---|---|---|---|---|"]
    keys = sorted({(r["scene"], r["prior"], r["start_rot_deg"], r["start_trans_cm"]) for r in conv})
    for k in keys:
        g = [r for r in conv if (r["scene"], r["prior"], r["start_rot_deg"], r["start_trans_cm"]) == k]
        m = lambda f: float(np.mean([f(r) for r in g]))  # noqa: E731
        L.append(f"| {k[0]} | {k[1]} | {k[2]} deg / {k[3]} cm | {m(lambda r: r['rot_err_deg']):.2f} | "
                 f"{m(lambda r: r['trans_err_cm']):.1f} | "
                 f"{m(lambda r: r['pos_err_lidar_cm']['x']):.1f} / {m(lambda r: r['pos_err_lidar_cm']['y']):.1f} / "
                 f"{m(lambda r: r['pos_err_lidar_cm']['z']):.1f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/calibration")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    perturbs = [(1.0, 3.0)] if a.quick else [(1.0, 3.0), (3.0, 8.0)]
    obs = study_observability()
    conv = study_convergence(perturbs, 1 if a.quick else 3)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "observability.json").write_text(json.dumps(obs, indent=2))
    (out / "convergence.json").write_text(json.dumps(conv, indent=2))
    (out / "report.md").write_text(report(obs, conv), encoding="utf-8")
    print(f"wrote {out}/report.md")


if __name__ == "__main__":
    main()
