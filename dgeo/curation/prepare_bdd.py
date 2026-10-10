"""BDD100K validation images (10,000 real dashcam frames with weather, time-of-day and scene labels) as a curation benchmark.

Source: https://huggingface.co/datasets/dgural/bdd100k, a FiftyOne copy of the official BDD100K validation split
(Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning", CVPR 2020; BSD-3 licence).

  python -m dgeo.curation.prepare_bdd --out data/bdd            # writes manifest.csv and downloads the images
  python -m dgeo.curation.embed_cli --manifest data/bdd/manifest.csv --models facebook/dinov2-small openai/clip-vit-base-patch32 --out data/bdd
  python -m dgeo.curation.run --emb dinov2=data/bdd/emb_dinov2-small.npz clip=data/bdd/emb_clip-vit-base-patch32.npz \
      --manifest data/bdd/manifest.csv --tags weather,time_of_day,scene --out results/curation_bdd

The manifest's `video` column is the first half of the file name: frames of one video share it, so train/test splits by
video never put the same clip on both sides.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = "https://huggingface.co/datasets/dgural/bdd100k/resolve/main/"


def label(entry) -> str:
    return (entry or {}).get("label") or "undefined"


def build_manifest(samples_json: Path) -> pd.DataFrame:
    samples = json.loads(samples_json.read_text(encoding="utf-8"))["samples"]
    rows = []
    for s in samples:
        name = Path(s["filepath"]).stem
        rows.append({"id": name, "path": "", "video": name.split("-")[0], "weather": label(s.get("weather")),
                     "time_of_day": label(s.get("timeofday")), "scene": label(s.get("scene")), "file": s["filepath"]})
    return pd.DataFrame(rows)


def fetch(args, tries: int = 4):
    """Download one image with a timeout and retries; a file is kept only if it starts like a JPEG."""
    url, dest = args
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            if data[:2] == bytes([255, 216]):          # JPEG magic number
                dest.write_bytes(data)
                return True
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bdd")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    sj = out / "samples.json"
    if not sj.exists():
        urllib.request.urlretrieve(BASE + "samples.json", sj)
    man = build_manifest(sj)
    if a.limit:
        man = man.sample(a.limit, random_state=0)
    man["path"] = [str(out / "images" / f"{i}.jpg") for i in man["id"]]
    jobs = [(BASE + f, Path(p)) for f, p in zip(man["file"], man["path"])]
    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(ex.map(fetch, jobs))
    man = man[ok].drop(columns="file")
    man.to_csv(out / "manifest.csv", index=False)
    print(f"{len(man)} images; weather {man.weather.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
