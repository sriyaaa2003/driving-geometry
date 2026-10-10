# driving-geometry

Three small research prototypes on driving data, built on one shared library (`dgeo/`) with embeddings, retrieval, camera models (pinhole and fisheye), pose-error metrics and bootstrap statistics.

## What I did
1. **Curation (`dgeo.curation`):** picking which images to keep. I wrote four samplers (random, k-means, k-center, hierarchical k-means) and scored them on rare-scene representation.
2. **Calibration (`dgeo.calibration`):** a study of which camera-position directions a drive actually constrains when calibrating a fisheye camera rig from the scene, using photometric consistency against LiDAR points on a synthetic rig.
3. **Drift (`dgeo.drift`):** a metric that checks whether a weather edit (fog, night, rain, blur) kept the road geometry in place, tested on edits with a known effect.

## Results (all on synthetic data)
- **Drift metric:** over 4 images and 30 edits each, dense optical-flow drift rises with injected shift, rotation and warps (Spearman rho 0.98). Fog, night, rain and blur edits sit 9 to 19 times below geometric corruptions of 2 px or more. The essential-matrix rotation signal is noisier (rho 0.64 to 0.83). Details: [results/drift/validation.md](results/drift/validation.md).
- **Calibration:** moving all cameras along the ground or walls does not change photometric consistency. After a coarse-to-fine solve on a straight path the leftover error is 8.7 cm along the travel direction (walls) or 25 cm sideways (open road), against under 1 cm along the ground normal. A curved path cuts the walls case to 4 cm. A road-plane prior on height, pitch and roll did not help. Report: [results/calibration_quick/report.md](results/calibration_quick/report.md).
- **Curation:** hierarchical k-means with leaf clusters about 8x the budget lifts rare-scene representation to about 2x random (random is 0.84x), close to greedy k-center (2.3x). With leaves equal to the budget it behaves like flat k-means (about 0.5x random).

## How it was built

```
dgeo/                shared: embed, retrieval, match (SIFT + RANSAC), geometry (cameras, poses), stats, synth
dgeo/curation/       samplers, metrics, run, localization
dgeo/calibration/    synthetic rig, loss, observability, solver, road prior, experiments
dgeo/drift/          metrics, synthetic edits, validation
tests/               11 tests
```

## Tech stack
Python, PyTorch, Transformers, NumPy, SciPy, OpenCV (SIFT + RANSAC), scikit-learn, pandas, matplotlib, pytest.
