"""Start the calibration runs on a deployed Modal app and return at once; results land in the 'dgeo-results'
volume (calibration_v2/<log>.csv), so a dropped connection on the laptop cannot lose them.

    modal deploy dgeo/calibration/modal_run_v2.py
    python scripts/launch_calibration.py --n-logs 60
    modal volume get dgeo-results /calibration_v2 results/calibration_v2_parts
"""
import argparse

import modal
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--n-logs", type=int, default=60)
ap.add_argument("--survey", default="results/calibration_v2/survey.csv")
a = ap.parse_args()
sv = pd.read_csv(a.survey).dropna(subset=["total_abs_heading_deg"])
curved = sv.sort_values("total_abs_heading_deg", ascending=False).log_id.tolist()[: a.n_logs // 2]
rest = [x for x in sv.log_id if x not in set(curved)]
step = max(1, len(rest) // (a.n_logs - len(curved)))
chosen = curved + rest[::step][: a.n_logs - len(curved)]
fn = modal.Function.from_name("dgeo-calibration-v2", "run_log")
for lid in chosen:
    fn.spawn(lid)
print("spawned", len(chosen), "logs")
