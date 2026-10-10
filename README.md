# driving-geometry

Four small studies on real driving data: dashcam images from [BDD100K](https://www.vis.xyz/bdd100k/) (the 10,000-image validation split, 9,992 downloaded) and camera + LiDAR drives from [Argoverse 2](https://www.argoverse.org/av2.html), built on one shared library (`dgeo/`) with embeddings, retrieval, camera geometry and bootstrap statistics.

## What I did
1. **Curation:** picking which images to keep. Using only DINOv2-small or CLIP ViT-B/32 embeddings (never the labels), I wrote five samplers (random, flat k-means, two hierarchical k-means variants, greedy k-center), picked 1%, 5% or 10% of the images, and scored how well the picks cover **rare scenes** (scene types that make up at most 5% of the data: parking lot, tunnel, gas station, unlabelled). Five seeds, bootstrap intervals.
2. **Drift:** a metric that checks whether a weather edit (fog, night, rain, blur) kept the road geometry in place. It compares an original and an edited image with dense optical flow, SIFT matches with RANSAC and an essential-matrix rotation, and is validated on 30 real daytime photos with edits of known effect applied on top (shifts, rotations, elastic warps, benign weather edits, pasted-in objects).
3. **Generated edits on Modal:** I ran depth-conditioned weather edits on the same 30 photos on a Modal A10G GPU (Stable Diffusion 1.5 + ControlNet depth, depth from Depth Anything V2 small): fog, night, rain and snow, 120 images in total. I then scored them with the drift metric.
4. **Calibration from real drives (Modal):** which directions of a camera's mounting position does a drive actually constrain? On 24 Argoverse 2 drives (240 segments of about 3 s) I took the front-centre camera as trusted and moved the front-left or front-right stereo camera away from its true position (random 0.5 degrees and 5 cm per axis). I then recovered it by projecting LiDAR points into both images and minimising the photometric difference between the two views, in 720 solves on a Modal T4 GPU.

## Results
Rare-scene lift at a 5% budget (share of the selection over share of the dataset; 1.0 is random):

| sampler | DINOv2 lift (95% CI) | DINOv2 coverage | CLIP lift | CLIP coverage |
|---|---|---|---|---|
| random | 0.93 (0.59 to 1.43) | 70% | 0.93 | 70% |
| flat k-means | 1.46 (1.00 to 2.08) | 85% | 1.45 | 85% |
| hierarchical k-means, sqrt-balanced | 2.75 (2.18 to 3.20) | 95% | 2.04 | 95% |
| hierarchical k-means, balanced | 4.49 (4.08 to 4.85) | 100% | 2.60 | 85% |
| **greedy k-center** | **4.83 (4.50 to 5.15)** | **100%** | **4.35** | **100%** |

- Embedding-based curation over-represents rare scenes 4 to 5 times at a 5% budget and covers every rare scene, against 70% for random.
- The two embedding models agree on what is rare (linear CKA 0.72, Spearman 0.73 between rarity scores).
- **Drift metric:** dense and sparse displacement rise with injected shift, rotation and warp (Spearman rho **0.98**). Benign edits have median dense drift 0.09 px against 2.63 px for corruptions of 2 px or more, a **28x** separation (9.7x for the sparse metric).
- Edge precision falls from 0.95 on benign edits to 0.89 when six objects are pasted in.

- **Generated edits (Modal):** all 120 generated edits drift more than 2 px. Median dense drift is 13.4 px for fog, 16.1 px for night, 11.6 px for rain and 15.5 px for snow, against 0.09 px for the benign injected edits. The metric clearly picks up that these edits moved the scene (cars, signs and lane markings shift). Mean edge precision is 0.63 to 0.74.

![generated edits](results/drift_generated/example.jpg)

*Top row: original, fog, night. Bottom row: rain, snow.*

- **Calibration (Argoverse 2):** rotations are recovered well. Mean absolute error goes from about 0.4 degrees to 0.03 (roll), 0.08 (pitch) and 0.05 (yaw), and 85% of solves end with all three rotations within 0.1 degrees. Sideways and vertical shifts go from about 3.4 cm to 1.0 cm. **The shift along the direction of travel is not recovered**: 3.4 cm before and 4.6 cm after. Straight drives and the 26 curved segments (at least 8 degrees of turning) give about the same errors. The Cramer-Rao uncertainty (0.1 to 0.2 cm for shifts) is far smaller than the errors that are actually left.

![calibration error by axis](results/calibration_av2/error_by_axis.png)

Tables: [results/calibration_av2/report.md](results/calibration_av2/report.md), [results/drift_generated/report.md](results/drift_generated/report.md), [results/curation_bdd/report.md](results/curation_bdd/report.md) and [results/drift/validation.md](results/drift/validation.md).

## How it was built

```
dgeo/embed.py, retrieval.py, match.py, geometry.py, stats.py     shared pieces
dgeo/curation/   prepare_bdd, embed_cli, samplers, metrics, run   study 1
dgeo/drift/      metrics, synth_edits, validate_metric            study 2
dgeo/drift/      modal_edit (ControlNet edits on Modal), score_pairs  study 3
dgeo/calibration/ av2 (loader), study (solver), modal_run, report  study 4
```

```bash
pip install -r requirements.txt && pip install -e . --no-deps
python -m dgeo.curation.prepare_bdd --out data/bdd
python -m dgeo.curation.run --emb dinov2=data/bdd/emb_dinov2-small.npz clip=data/bdd/emb_clip-vit-base-patch32.npz --manifest data/bdd/manifest.csv --tags weather,time_of_day,scene --out results/curation_bdd
python -m dgeo.drift.validate_metric --images data/bdd/drift_images
modal run dgeo/drift/modal_edit.py --images data/bdd/drift_images --out data/edited
modal run dgeo/calibration/modal_run.py --n-logs 24 --out results/calibration_av2
python -m dgeo.calibration.report
python -m dgeo.drift.score_pairs --orig data/bdd/drift_images --edited data/edited/fog --out results/drift_generated/fog
pytest
```

9 tests.

## Tech stack
Python, PyTorch, Transformers (DINOv2, CLIP, Depth Anything V2), Diffusers (Stable Diffusion + ControlNet), Modal (GPU), pyarrow, NumPy, SciPy, OpenCV (SIFT + RANSAC), scikit-learn, pandas, matplotlib, pytest.

## Data
Yu et al., *BDD100K*, CVPR 2020 (BSD-3), via the FiftyOne copy at `dgural/bdd100k`. Wilson et al., *Argoverse 2*, NeurIPS Datasets and Benchmarks 2021 (CC BY-NC-SA 4.0), streamed from the open S3 bucket. The data is not redistributed.
