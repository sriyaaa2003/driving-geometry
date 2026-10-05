"""Zip everything worth keeping from results/ (no big files) into results/pack_<timestamp>.zip so it can go home on
a USB stick or be emailed to yourself -- no GitHub login needed on the lab computer.

    python scripts/pack_results.py
"""
import time
import zipfile
from pathlib import Path

root = Path("results")
name = root / f"pack_{time.strftime('%Y%m%d_%H%M%S')}.zip"
n = 0
with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in {".md", ".json", ".csv", ".png", ".txt", ".log"} and p.stat().st_size < 20e6:
            z.write(p, p.relative_to(root.parent))
            n += 1
print(f"packed {n} files -> {name}  ({name.stat().st_size / 1e6:.2f} MB)")
