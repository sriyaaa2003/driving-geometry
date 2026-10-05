"""Run this FIRST on the lab computer. Prints what the machine can do and writes results/lab_check.json.

    python scripts/lab_check.py
"""
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

report = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": {}, "gpu": None}
for mod in ["numpy", "scipy", "sklearn", "pandas", "cv2", "torch", "transformers", "gsplat", "diffusers", "zod"]:
    try:
        m = importlib.import_module(mod)
        report["packages"][mod] = getattr(m, "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        report["packages"][mod] = f"MISSING ({type(e).__name__})"

try:
    import torch

    report["torch_cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        report["gpu"] = {"name": p.name, "vram_gb": round(p.total_memory / 1e9, 1), "cuda": torch.version.cuda,
                         "capability": f"{p.major}.{p.minor}"}
except Exception as e:  # noqa: BLE001
    report["torch_error"] = str(e)

if shutil.which("nvidia-smi"):
    report["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()

report["disk_free_gb"] = round(shutil.disk_usage(".").free / 1e9, 1)


def _render(camera_model, focal):
    import torch
    from gsplat import rasterization

    n = 50
    means = torch.randn(n, 3, device="cuda") * 0.3 + torch.tensor([0, 0, 3.0], device="cuda")
    quats = torch.zeros(n, 4, device="cuda")
    quats[:, 0] = 1
    K = torch.tensor([[focal, 0, 64.0], [0, focal, 64.0], [0, 0, 1.0]], device="cuda")[None]
    kw = {} if camera_model == "pinhole" else {"camera_model": camera_model}
    out, _, _ = rasterization(means, quats, torch.full((n, 3), 0.1, device="cuda"),
                              torch.full((n,), 0.8, device="cuda"), torch.rand(n, 3, device="cuda"),
                              torch.eye(4, device="cuda")[None], K, 128, 128, **kw)
    return tuple(out.shape)


# tiny gsplat renders, only if everything needed is present
if report.get("torch_cuda_available") and not str(report["packages"]["gsplat"]).startswith("MISSING"):
    for name, model, focal in [("gsplat_pinhole_render", "pinhole", 100.0), ("gsplat_fisheye_render", "fisheye", 60.0)]:
        try:
            report[name] = f"ok {_render(model, focal)}"
        except Exception as e:  # noqa: BLE001
            report[name] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"

Path("results").mkdir(exist_ok=True)
Path("results/lab_check.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print("\nSaved results/lab_check.json - send me this output (or the file).")
