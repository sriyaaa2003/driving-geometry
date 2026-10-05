# driving-geometry

Three connected research prototypes on driving data: **curating** it with foundation-model embeddings,
**calibrating** a camera rig from the scene itself, and **measuring** whether generated weather edits keep the
geometry. They share one library (`dgeo/`): embeddings, retrieval, camera models (pinhole and fisheye),
pose-error metrics and bootstrap statistics.

| # | Project | Question | Status |
|---|---|---|---|
| 1 | `dgeo.curation` | Which images are worth keeping? Score curation by held-out scenario tags and by cross-dataset localization recall under a database-size budget. | Algorithms, metrics and CLI tested on synthetic data. **Not yet run on real data.** |
| 2 | `dgeo.calibration` | Which extrinsic directions does a drive actually constrain when calibrating a fisheye rig without a target? | CPU study done on a synthetic rig. Gaussian-splatting version written, **never run on a GPU**. |
| 3 | `dgeo.drift` | Did a weather edit keep the road where it was? A geometry-drift metric, validated on edits with known effect. | Metric validated on synthetic images. ControlNet editing script written, **never run**. |

Everything below labelled *synthetic* is a property of the method on a controlled scene, not a result on a real vehicle.

## Findings so far (synthetic data)

**Project 3 — the drift metric works on controlled edits.** Over 4 images and 30 edits each: dense optical-flow drift
rises with injected shift, rotation and elastic warps (Spearman rho 0.98), and benign fog/night/rain/blur edits sit
9–19x below geometric corruptions of 2 px or more. Weaknesses: the essential-matrix rotation signal is noisier
(rho 0.64–0.83), and `edge_precision` barely separates a single pasted-in object from benign edits (0.92 vs 0.88).
See [results/drift/validation.md](results/drift/validation.md). Needs repeating on real photos (`--images`).

**Project 2 — photometric consistency cannot see translation along the surfaces.** A common shift of all cameras
along a direction tangent to the ground and walls keeps every LiDAR point on its surface, so multi-view
consistency does not change. On a straight path the residual error after a coarse-to-fine solve is 8.7 cm along
the direction of travel (walls) or 25 cm sideways (open road), against under 1 cm along the ground normal. A
curved path cuts the walls case to 4 cm, because the heading differs between frames.
**Negative result:** a road-plane prior on camera height, pitch and roll did not help (it only constrains
directions that are already well observed). Fixing the weak directions needs scene structure or a geometric
term. Linear (Cramer-Rao) uncertainty is optimistic here, because it ignores systematic model error.
Quick run (1 trial per cell): [results/calibration_quick/report.md](results/calibration_quick/report.md); the full
run is `python -m dgeo.calibration.experiments`.

**Project 1 — a balanced hierarchy only works with fine leaves.** On synthetic long-tailed embeddings,
hierarchical k-means with leaf clusters about 8x the budget lifts rare-scene representation to about 2x random
(random 0.84x), close to greedy k-center (2.3x). With leaves equal to the budget it degenerates to flat k-means,
which under-samples rare scenes (about 0.5x random in the 3-seed demo run). Real-data numbers are still to come.

## Quick start (CPU, no data needed)

```bash
pip install -r requirements.txt          # plus a PyTorch build, see LAB_GUIDE.md
pip install -e . --no-deps
python -m pytest tests -q                # 11 tests, about 15 s
python -m dgeo.drift.validate_metric     # drift-metric validation
python -m dgeo.calibration.experiments --quick
python -m dgeo.curation.run --demo
python -m dgeo.curation.localization --demo
```

## Real data

* **ZOD** (Zenseact Open Dataset, CC BY-SA 4.0): `pip install zod`, download a small subset with `zod download`,
  then `python -m dgeo.curation.prepare_zod inspect ...` (prints what the SDK really exposes), `manifest`,
  `python -m dgeo.curation.embed_cli ...`, `python -m dgeo.curation.run ...`. The tag names in
  `prepare_zod.py` are candidates and **have not been checked against real ZOD files**.
* **KITTI-360** (fisheye + LiDAR + calibration, CC BY-NC-SA 3.0, registration required) is the intended source for
  the calibration project. A loader is not written yet.
* Localization ground truth for ZOD frames is unverified; MSLS is the fallback.

## GPU parts (written, not yet run)

```bash
python scripts/lab_check.py              # what the machine can do
python scripts/lab_run.py gs-smoke       # Gaussian-splatting calibration smoke test on synthetic data
python -m dgeo.drift.edit_controlnet --images data/imgs --out data/edited      # depth-conditioned weather edits
```

See [LAB_GUIDE.md](LAB_GUIDE.md) for the exact lab-computer procedure.

## Layout

```
dgeo/                shared: embed, retrieval, match (SIFT + RANSAC), geometry (cameras, poses), stats, synth
dgeo/curation/       samplers (random, k-means, k-center, hierarchical k-means), metrics, run, localization
dgeo/calibration/    scene (synthetic rig), core (loss, observability, solver, road prior), experiments, gs_calib
dgeo/drift/          metrics, synth_edits, validate_metric, score_pairs, edit_controlnet
scripts/             lab_check, lab_run, pack_results
tests/               11 tests
```

## Honest limitations

* The calibration study's scene is analytic planes with a synthetic texture; real scenes have edges, poles and
  moving objects that would change which directions are observable.
* Project 3 conditions on monocular depth, a stand-in for occupancy or LiDAR-projected depth.
* Classical SIFT/RANSAC is the matcher on CPU; LightGlue is not wired in yet.
* ZOD and KITTI-360 licences must be respected: credit the datasets and keep the data out of this repository.
