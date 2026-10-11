# Drift metric v2: checked against independent judges

30 real BDD100K photos. All pairs are scored at the same width (480 px), generated or synthetic. Judges that do not use optical flow: semantic-segmentation agreement (SegFormer, Cityscapes) and monocular-depth rank correlation (Depth Anything V2).

## 1. Detecting geometry that really moved (synthetic edits with known effect)

Positives: shifts of at least 2 px, rotations of at least 0.5 deg, elastic warps of at least 2 px. Negatives: fog, night, rain and blur of three strengths. AUC 1.0 = perfect, 0.5 = chance. 95% bootstrap CI over images.

| measure | AUC |
|---|---|
| drift: dense flow (px) | 0.991 [0.975, 1.000] |
| drift: sparse SIFT (px) | 1.000 [0.999, 1.000] |
| drift: essential-matrix rotation (deg) | 0.859 [0.831, 0.886] |
| edge precision | 0.752 [0.743, 0.761] |
| segmentation pixel agreement | 0.179 [0.136, 0.212] |
| segmentation IoU, road+building+sky+... | 0.267 [0.222, 0.307] |
| depth rank correlation | 0.588 [0.524, 0.653] |

## 2. What the generators do (medians over images, prompts and seeds)

| generator | dense drift (px) | sparse drift (px) | edge precision | seg. pixel agreement | seg. structure IoU | depth rank corr. |
|---|---|---|---|---|---|---|
| depth ControlNet, neutral prompt (generator floor) | 10.94 | 141.44 | 0.69 | 0.74 | 0.88 | 0.98 |
| img2img 0.5, no control | 1.83 | 10.01 | 0.86 | 0.77 | 0.86 | 0.97 |
| depth ControlNet | 13.18 | 151.83 | 0.71 | 0.64 | 0.87 | 0.97 |
| canny ControlNet | 10.81 | 43.65 | 0.89 | 0.65 | 0.84 | 0.98 |
| depth + canny ControlNet | 7.42 | 1.38 | 0.93 | 0.75 | 0.89 | 0.99 |

Weather prompt vs the generator floor (depth ControlNet, dense drift in px, mean over images):

| prompt | with weather prompt | neutral prompt (floor) | difference [95% CI] |
|---|---|---|---|
| fog | 14.40 | 13.30 | 1.10 [-0.42, 2.56] |
| night | 20.02 | 13.30 | 6.72 [4.13, 9.41] |
| rain | 15.41 | 13.30 | 2.12 [-1.39, 7.26] |
| snow | 20.52 | 13.30 | 7.22 [2.93, 12.66] |

## 3. On generated images, does drift agree with the independent judges?

Spearman correlation across all generated edits between dense drift and each judge (a judge column is flipped so that higher = more change).

| judge | rho with dense drift |
|---|---|
| drift: sparse SIFT (px) | 0.49 |
| drift: essential-matrix rotation (deg) | 0.43 |
| edge precision | 0.22 |
| segmentation pixel agreement | 0.40 |
| segmentation IoU, road+building+sky+... | 0.06 |
| depth rank correlation | -0.09 |
