# driving-geometry

Two small studies on real dashcam data from [BDD100K](https://www.vis.xyz/bdd100k/) (the 10,000-image validation split, 9,992 downloaded), built on one shared library (`dgeo/`) with embeddings, retrieval, camera geometry and bootstrap statistics.

## What I did
1. **Curation:** picking which images to keep. Using only DINOv2-small or CLIP ViT-B/32 embeddings (never the labels), I wrote five samplers (random, flat k-means, two hierarchical k-means variants, greedy k-center), picked 1%, 5% or 10% of the images, and scored how well the picks cover **rare scenes** (scene types that make up at most 5% of the data: parking lot, tunnel, gas station, unlabelled). Five seeds, bootstrap intervals.
2. **Drift:** a metric that checks whether a weather edit (fog, night, rain, blur) kept the road geometry in place. It compares an original and an edited image with dense optical flow, SIFT matches with RANSAC and an essential-matrix rotation, and is validated on 30 real daytime photos with edits of known effect applied on top (shifts, rotations, elastic warps, benign weather edits, pasted-in objects).

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

Tables: [results/curation_bdd/report.md](results/curation_bdd/report.md) and [results/drift/validation.md](results/drift/validation.md).

## How it was built

```
dgeo/embed.py, retrieval.py, match.py, geometry.py, stats.py     shared pieces
dgeo/curation/   prepare_bdd, embed_cli, samplers, metrics, run   study 1
dgeo/drift/      metrics, synth_edits, validate_metric            study 2
```

```bash
pip install -r requirements.txt && pip install -e . --no-deps
python -m dgeo.curation.prepare_bdd --out data/bdd
python -m dgeo.curation.run --emb dinov2=data/bdd/emb_dinov2-small.npz clip=data/bdd/emb_clip-vit-base-patch32.npz --manifest data/bdd/manifest.csv --tags weather,time_of_day,scene --out results/curation_bdd
python -m dgeo.drift.validate_metric --images data/bdd/drift_images
pytest
```

9 tests.

## Tech stack
Python, PyTorch, Transformers (DINOv2, CLIP), NumPy, SciPy, OpenCV (SIFT + RANSAC), scikit-learn, pandas, matplotlib, pytest.

## Data
Yu et al., *BDD100K*, CVPR 2020 (BSD-3), via the FiftyOne copy at `dgural/bdd100k`. The images are not redistributed.
