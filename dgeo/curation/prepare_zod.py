"""Zenseact Open Dataset (ZOD) helpers.  NOT yet verified against real ZOD files -- run `inspect` first.

  pip install zod
  zod download --help                      # download the 'mini' Frames subset first (small)
  python -m dgeo.curation.prepare_zod inspect --root data/zod --version mini
  python -m dgeo.curation.prepare_zod manifest --root data/zod --version mini --out data/zod_manifest.csv

`inspect` prints the attributes the SDK really exposes for one frame, so the candidate tag names below can
be corrected against reality. `manifest` writes id, path and every candidate tag the SDK provides.
"""
from __future__ import annotations

import argparse

CANDIDATE_TAGS = [
    "time_of_day", "scraped_weather", "weather", "country_code", "country", "road_type",
    "road_condition", "num_lane_instances", "num_vehicles", "num_vru", "ego_speed",
    "latitude", "longitude", "lat", "lon", "collection_car", "start_time",
]


def _frames(root: str, version: str):
    from zod import ZodFrames

    return ZodFrames(dataset_root=root, version=version)


def _frame_ids(zf):
    for name in ("get_all_ids", "get_all_frame_ids"):
        if hasattr(zf, name):
            return sorted(getattr(zf, name)())
    from zod import constants

    return sorted(set(zf.get_split(constants.TRAIN)) | set(zf.get_split(constants.VAL)))


def _image_path(frame):
    from zod.constants import Anonymization

    return str(frame.info.get_key_camera_frame(Anonymization.DNAT).filepath)


def inspect(root: str, version: str):
    zf = _frames(root, version)
    ids = _frame_ids(zf)
    print(f"{len(ids)} frames; first id: {ids[0]}")
    frame = zf[ids[0]]
    print("frame attributes:", [a for a in dir(frame) if not a.startswith("_")])
    info = frame.info
    print("info attributes:")
    for a in dir(info):
        if a.startswith("_"):
            continue
        try:
            v = getattr(info, a)
        except Exception as e:  # noqa: BLE001
            v = f"<error {e}>"
        if not callable(v):
            print(f"  {a} = {str(v)[:100]}")
    try:
        print("key camera path:", _image_path(frame))
    except Exception as e:  # noqa: BLE001
        print("could not get key camera path:", e)


def manifest(root: str, version: str, out: str, limit: int = 0):
    import pandas as pd

    zf = _frames(root, version)
    ids = _frame_ids(zf)
    if limit:
        ids = ids[:limit]
    rows = []
    for fid in ids:
        frame = zf[fid]
        row = {"id": str(fid), "path": _image_path(frame)}
        for name in CANDIDATE_TAGS:
            v = getattr(frame.info, name, None)
            if isinstance(v, (str, int, float, bool)):
                row[name] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows); tag columns: {[c for c in rows[0] if c not in ('id', 'path')]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "manifest"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", default="mini", choices=["mini", "full"])
    ap.add_argument("--out", default="data/zod_manifest.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "inspect":
        inspect(a.root, a.version)
    else:
        manifest(a.root, a.version, a.out, a.limit)


if __name__ == "__main__":
    main()
