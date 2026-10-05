"""Embed every image in a manifest with one or more foundation models, writing data/emb_<name>.npz.

  python -m dgeo.curation.embed_cli --manifest data/manifest.csv --models facebook/dinov2-small \
      openai/clip-vit-base-patch32 --out data --batch-size 64
Manifest needs columns 'id' and 'path'. Finished models are skipped on re-run (cache), so it is safe to
resume after a crash or a lab-session timeout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dgeo.embed import embed_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--models", nargs="+", default=["facebook/dinov2-small"])
    ap.add_argument("--out", default="data")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (for a quick test)")
    ap.add_argument("--suffix", default="", help="added to output names, e.g. _zod")
    a = ap.parse_args()
    man = pd.read_csv(a.manifest)
    if a.limit:
        man = man.head(a.limit)
    paths, ids = man["path"].tolist(), man["id"].astype(str).tolist()
    for m in a.models:
        short = m.split("/")[-1]
        cache = Path(a.out) / f"emb_{short}{a.suffix}.npz"
        print(f"[embed_cli] {m} -> {cache}")
        embed_paths(paths, m, a.batch_size, cache=cache, ids=ids)


if __name__ == "__main__":
    main()
