"""One-command stages for the lab computer. Each stage logs to results/logs/<stage>.log and can be re-run.

    python scripts/lab_run.py check      # hardware + package check (run first)
    python scripts/lab_run.py studies    # CPU studies: drift-metric validation, calibration study, curation demo
    python scripts/lab_run.py gs-smoke   # Gaussian-splatting calibration smoke test on synthetic data (GPU)
    python scripts/lab_run.py all
"""
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
STAGES = {
    "check": [[PY, "scripts/lab_check.py"]],
    "studies": [
        [PY, "-m", "dgeo.drift.validate_metric", "--out", "results/drift"],
        [PY, "-m", "dgeo.calibration.experiments", "--out", "results/calibration"],
        [PY, "-m", "dgeo.curation.run", "--demo", "--out", "results/demo_curation"],
    ],
    "gs-smoke": [
        [PY, "-m", "dgeo.calibration.export_synth", "--path", "curve", "--camera", "pinhole",
         "--out", "data/synth_curve_pinhole.npz"],
        [PY, "-m", "dgeo.calibration.gs_calib", "--data", "data/synth_curve_pinhole.npz", "--steps", "3000",
         "--out", "results/gs_calib/curve_pinhole"],
        [PY, "-m", "dgeo.calibration.gs_calib", "--data", "data/synth_curve_pinhole.npz", "--steps", "3000",
         "--freeze-extrinsics", "--out", "results/gs_calib/control_frozen"],
    ],
}


def run_stage(name):
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    log = Path(f"results/logs/{name}.log")
    with log.open("w", encoding="utf-8") as f:
        for cmd in STAGES[name]:
            line = f"\n$ {' '.join(cmd)}\n"
            print(line, flush=True)
            f.write(line)
            t = time.time()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for out in p.stdout:
                print(out, end="")
                f.write(out)
            p.wait()
            f.write(f"\n[exit {p.returncode}, {time.time() - t:.0f}s]\n")
            if p.returncode != 0:
                print(f"STAGE FAILED: {' '.join(cmd)}\nlog: {log}")
                return False
    return True


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "check"
    names = list(STAGES) if which == "all" else [which]
    ok = all(run_stage(n) for n in names)
    print("\nDONE" if ok else "\nSOME STAGES FAILED - send me results/logs/")
