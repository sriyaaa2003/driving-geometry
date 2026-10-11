# driving-geometry

Four small studies on real driving data: dashcam images from [BDD100K](https://www.vis.xyz/bdd100k/) (the 10,000-image validation split, 9,992 downloaded) and camera + LiDAR drives from [Argoverse 2](https://www.argoverse.org/av2.html). They share one library (`dgeo/`) with embeddings, retrieval, camera geometry and bootstrap statistics. The GPU work ran on Modal.

## What I did

**1. Curation: picking which images to keep.**
- Using only DINOv2-small or CLIP ViT-B/32 embeddings (never the labels), I picked 1% or 5% of the 9,992 images with eight methods: random, flat k-means, SemDeDup, D4, SSL Prototypes (keep the hardest examples), two hierarchical k-means variants and greedy k-center.
- I scored how many rare scene types (at most 5% of the data: parking lot, tunnel, gas station) each pick covers. "Undefined" labels are not counted as scenes.
- Then I added a downstream test: split the images 70/30, let each method pick 5% of the 70%, train a logistic-regression classifier on the picked images (using the other embedding model's features), and score it on the held-out 30% for scene, weather, time of day and weather-plus-time combinations. Three splits, three seeds.

**2. Drift: a "did the scene move?" metric.**
- It compares an original and an edited photo with dense optical flow, SIFT matches with RANSAC, an essential-matrix rotation and edge precision.
- I tested it on 30 real photos with edits of known effect (shifts, rotations, elastic warps, fog, night, rain, blur, pasted-in objects) and compared it with two independent judges that do not use optical flow: semantic-segmentation agreement (SegFormer) and monocular-depth rank correlation (Depth Anything V2).

**3. Generated weather edits on Modal.**
- On a Modal A10G GPU I generated fog, night, rain and snow edits of the same 30 photos with five generator settings: img2img with no structural control, depth ControlNet, canny ControlNet, depth + canny ControlNet, and a depth ControlNet given a neutral prompt (to measure how much the generator moves things when asked for no change). Two seeds each, 1,920 scored images in total, all scored at the same 480 px width.

**4. Camera calibration from real drives on Modal.**
- On 57 Argoverse 2 drives (286 segments of about 3 s, half from the most-turning drives) I took the front-centre camera as trusted and moved a front stereo camera away from its reference position. I recovered it by projecting LiDAR points into both images and minimising the photometric difference between the two views.
- I also ran: starting at the reference (no perturbation), starts from 0.25 to 2 degrees and 2.5 to 20 cm away, LiDAR points restricted to near, mid or far, segments grouped by how much the car turned, and a classical baseline (SIFT matches + PnP with LiDAR depth).

## Results

**Curation, rare-scene lift at a 5% budget (DINOv2; 1.0 is random)**

| method | rare-scene lift (95% CI) | rare scene types covered |
|---|---|---|
| random | 1.05 (0.49 to 2.12) | 56% |
| flat k-means | 1.32 (0.79 to 1.88) | 78% |
| SemDeDup | 5.03 (4.46 to 5.93) | 100% |
| D4 | 4.37 (4.14 to 4.71) | 100% |
| SSL Prototypes (hardest) | 6.77 (5.96 to 7.45) | 100% |
| hierarchical k-means, sqrt-balanced | 2.85 (1.99 to 3.92) | 89% |
| hierarchical k-means, balanced | 5.84 (5.52 to 6.18) | 100% |
| greedy k-center | 5.84 (5.14 to 6.45) | 100% |

- With CLIP, the same methods give lifts of 3.1 to 5.3 and random 1.05. The two models agree on what is rare (linear CKA 0.72, Spearman 0.73).
- Downstream, on the weather-plus-time-of-day combinations (rare combinations have 635 test images on average), recall on the rare combinations is 2.8% for a classifier trained on a random 5%, 9.6% for hierarchical balanced k-means, 5.4% to 6.0% for SemDeDup, D4, SSL Prototypes and k-center, and 33% when training on all the data. Macro-F1 goes from 0.115 (random) to 0.132 (hierarchical balanced k-means); all data gives 0.293.
- For weather alone macro-F1 goes from 0.231 (random) to 0.29 (SemDeDup), and for time of day from 0.636 to 0.671. Scene macro-F1 is about 0.27 to 0.29 for every method. Recall on the rare scene types is 0 for every sampler and 2.7% for all data; the held-out set has only about 23 images of them.

![drift v2](results/drift_v2/drift_v2.png)

**Drift metric**
- Telling moved geometry from appearance-only edits (shifts of 2 px or more, rotations of 0.5 degrees or more and elastic warps of 2 px or more against fog, night, rain and blur): AUC 0.991 for dense flow drift, 1.000 for the sparse SIFT version, 0.859 for the essential-matrix rotation, 0.752 for edge precision, 0.588 for depth correlation and 0.179 for segmentation agreement.
- Generator settings (median dense drift in px / sparse drift in px): depth ControlNet with a neutral prompt 10.9 / 141; img2img at strength 0.5 1.8 / 10; depth ControlNet 13.2 / 152; canny ControlNet 10.8 / 44; depth + canny 7.4 / 1.4.
- With the depth ControlNet, the weather prompt adds on top of the neutral-prompt level: fog +1.1 px (95% CI -0.4 to 2.6), night +6.7 (4.1 to 9.4), rain +2.1 (-1.4 to 7.3), snow +7.2 (2.9 to 12.7).
- Across all generated edits, dense drift correlates with segmentation change at rho 0.40, with essential-matrix rotation at 0.43, with sparse drift at 0.49, with edge precision at 0.22, and with depth correlation at -0.09.

![calibration v2](results/calibration_v2/calibration_v2.png)

**Calibration (Argoverse 2, 57 drives)**
- Starting 0.5 degrees and 5 cm away, the rotations come back to 0.03 (roll), 0.06 (pitch) and 0.05 (yaw) degrees from the reference, and 83% of starts end with all three rotations within 0.1 degrees. Left and up shifts end at 1.1 and 1.2 cm, forward shift at 4.7 cm (initial errors 3.1, 1.9 and 3.3 cm).
- Starting exactly at the reference, the solver moves forward by a median 4.4 cm (left 0.9 cm, up 1.1 cm, rotations 0.02 to 0.04 degrees). The classical PnP baseline finds the same forward offset (3.5 cm) and is better on forward shift (4.1 against 4.5 cm); the photometric solver is better on roll (0.018 against 0.049 degrees), left shift (0.9 against 1.6 cm) and up shift (1.1 against 1.2 cm). The forward offset is 4.2 cm at standstill and 5.6 cm above 10 m/s.
- The reference extrinsics differ between logs by about 0.2 degrees and 0.4 cm (standard deviation, 80 distinct values for stereo_front_left across 141 logs).
- Basin of convergence: 90% of starts converge within 0.25 degrees and 2.5 cm, 83% at 0.5 degrees and 5 cm, 31% at 1 degree and 10 cm and 9% at 2 degrees and 20 cm.
- Depth of the LiDAR points: forward-shift error is 2.7 cm with points at 4 to 15 m, 5.3 cm at 15 to 30 m and 8.8 cm at 30 to 50 m. Rotation errors are 0.02 to 0.08 degrees at every depth.
- Turning: forward-shift error is 4.7 cm for segments with under 3 degrees of heading change, 4.6 cm at 3 to 10, 5.1 cm at 10 to 25 and 3.2 cm above 25 degrees (Spearman -0.13).
- The Cramer-Rao standard deviations (0.003 to 0.004 degrees, 0.09 to 0.19 cm) are much smaller than the spread of the actual errors across segments (0.13 to 0.47 degrees, 2 to 4 cm).

Full tables: [curation coverage](results/curation_v2/coverage_report.md), [curation downstream](results/curation_v2/downstream_report.md), [drift v2](results/drift_v2/report.md), [calibration v2](results/calibration_v2/report.md). The first versions of the studies are in `results/curation_bdd`, `results/drift`, `results/drift_generated` and `results/calibration_av2`.

## How it was built

```
dgeo/embed.py, retrieval.py, match.py, geometry.py, stats.py        shared pieces
dgeo/curation/    prepare_bdd, embed_cli, samplers, baselines, downstream, run_v2     study 1
dgeo/drift/       metrics, synth_edits, validate_metric, modal_validate, analyse_v2   studies 2 and 3
dgeo/calibration/ av2 (loader), study (solver), baseline (PnP), modal_run_v2, analyse_v2   study 4
scripts/          launch_calibration (starts the Modal runs and returns)
```

```bash
pip install -r requirements.txt && pip install -e . --no-deps
python -m dgeo.curation.prepare_bdd --out data/bdd
python -m dgeo.curation.run_v2 --emb dinov2=data/bdd/emb_dinov2-small.npz clip=data/bdd/emb_clip-vit-base-patch32.npz --manifest data/bdd/manifest.csv --seeds 3 --splits 3
modal run dgeo/drift/modal_validate.py --images data/bdd/drift_images --out results/drift_v2
python -m dgeo.drift.analyse_v2
modal run dgeo/calibration/modal_run_v2.py::survey
modal deploy dgeo/calibration/modal_run_v2.py && python scripts/launch_calibration.py --n-logs 60
python -m dgeo.calibration.analyse_v2
pytest
```

## Tech stack
Python, PyTorch, Transformers (DINOv2, CLIP, SegFormer, Depth Anything V2), Diffusers (Stable Diffusion 1.5 + ControlNet depth and canny), Modal (A10G and T4 GPUs), pyarrow, NumPy, SciPy, OpenCV (optical flow, SIFT, RANSAC, PnP), scikit-learn, pandas, matplotlib, pytest.

## Data
Yu et al., *BDD100K*, CVPR 2020 (BSD-3), via the FiftyOne copy at `dgural/bdd100k`. Wilson et al., *Argoverse 2*, NeurIPS Datasets and Benchmarks 2021 (CC BY-NC-SA 4.0), streamed from the open S3 bucket. The data is not redistributed.
