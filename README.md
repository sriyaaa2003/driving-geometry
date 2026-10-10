# driving-geometry

Two small studies on real dashcam data from [BDD100K](https://www.vis.xyz/bdd100k/) (the 10,000-image validation split, 9,992 downloaded;
real photos with weather, time-of-day and scene labels).

## 1. Which images are worth keeping? Data curation with foundation-model embeddings

Pick a budget of 1%, 5% or 10% of the images using only their DINOv2-small or CLIP ViT-B/32 embeddings (never the labels), then ask how well
the picks cover the **rare scenarios**: scene types that make up at most 5% of the data (parking lot, tunnel, gas station, unlabelled;
7 to 53 images each). Five samplers, five seeds; intervals are bootstrap.

Rare-scene lift at a 5% budget (share of the selection over share of the dataset; 1.0 = random) and coverage of rare scenes:

| sampler | DINOv2 lift [95% CI] | DINOv2 coverage | CLIP lift | CLIP coverage |
|---|---|---|---|---|
| random | 0.93 [0.59, 1.43] | 70% | 0.93 | 70% |
| flat k-means | 1.46 [1.00, 2.08] | 85% | 1.45 | 85% |
| hierarchical k-means, sqrt-balanced | 2.75 [2.18, 3.20] | 95% | 2.04 | 95% |
| hierarchical k-means, balanced | 4.49 [4.08, 4.85] | 100% | 2.60 | 85% |
| **greedy k-center** | **4.83 [4.50, 5.15]** | **100%** | **4.35** | **100%** |

- **Embedding-based curation finds rare scenes much better than random**: 4 to 5 times over-represented at 5%, and every rare scene covered
  against 70% for random.
- **It is not free of trade-offs.** At a 1% budget the balanced samplers work (lift about 10) but the sqrt-balanced one and flat k-means
  fail (lift under 1); k-center chases outliers, so some of what it picks may simply be odd rather than rare. The weather tag has only one
  rare value (fog, 13 images), so its numbers are too noisy to rank samplers.
- **The two models mostly agree on what is rare** (linear CKA 0.72, Spearman 0.73 between rarity scores) but agree on the top 5% only 21%.

```bash
pip install -r requirements.txt && pip install -e . --no-deps
python -m dgeo.curation.prepare_bdd --out data/bdd         # manifest + 9,992 images (about 1.3 GB)
python -m dgeo.curation.embed_cli --manifest data/bdd/manifest.csv --models facebook/dinov2-small openai/clip-vit-base-patch32 --out data/bdd
python -m dgeo.curation.run --emb dinov2=data/bdd/emb_dinov2-small.npz clip=data/bdd/emb_clip-vit-base-patch32.npz \
    --manifest data/bdd/manifest.csv --tags weather,time_of_day,scene --out results/curation_bdd
```

## 2. Did a weather edit keep the road where it was? A geometry-drift metric

Generative weather edits can move lane lines and cars. The metric compares an original and an edited image with dense optical flow, sparse
SIFT matches with RANSAC and an essential-matrix rotation. It is validated on **30 real BDD100K daytime photos** with edits of known
effect applied on top (shifts, rotations and elastic warps of injected size; benign fog, night, rain and blur; pasted-in objects):

- **Dose-response**: dense and sparse displacement rise with injected shift, rotation and warp (Spearman rho **0.98**); the essential-matrix
  rotation is weaker (0.36 to 0.82).
- **Specificity**: benign edits have median dense drift 0.09 px against 2.63 px for corruptions of 2 px or more, a **28x** separation (9.7x
  for the sparse metric).
- **Hallucination**: edge precision falls from 0.95 on benign edits to 0.89 when six objects are pasted in, but one pasted object (0.97) is not
  detected.

Edits are synthetic by design (that is how the truth is known); the photos are real. Results are in `results/drift/`. This validates the
metric only: no generative edit model was run.

```bash
python -m dgeo.drift.validate_metric --images data/bdd/drift_images
pytest        # 9 tests: samplers, curation metrics, retrieval, camera geometry, edit and drift functions (on small hand-built arrays)
```

## Layout

```
dgeo/embed.py, retrieval.py, match.py, geometry.py, stats.py      shared pieces
dgeo/curation/   prepare_bdd, embed_cli, samplers, metrics, run   study 1
dgeo/drift/      metrics, synth_edits, validate_metric             study 2
```

Data: Yu et al., "BDD100K", CVPR 2020 (BSD-3), via the FiftyOne copy at `dgural/bdd100k`; the images are not redistributed here.
