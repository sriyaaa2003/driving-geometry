# Geometry-drift metric validation

Images with KNOWN edits. A trustworthy metric must (1) rise with injected geometric corruption, (2) stay low for benign appearance edits, (3) notice hallucinated structure.

Images: 4, edits per image: 30.

## 1. Dose-response (Spearman rho between injected magnitude and metric; 1.0 = perfect)

| corruption : metric | rho |
|---|---|
| elastic:dense_disp_p90 | 0.98 |
| elastic:dense_disp_px | 0.98 |
| elastic:rel_rot_deg | 0.64 |
| elastic:sparse_disp_px | 0.98 |
| rotate:dense_disp_p90 | 0.98 |
| rotate:dense_disp_px | 0.98 |
| rotate:rel_rot_deg | 0.83 |
| rotate:sparse_disp_px | 0.98 |
| shift:dense_disp_p90 | 0.98 |
| shift:dense_disp_px | 0.98 |
| shift:rel_rot_deg | 0.77 |
| shift:sparse_disp_px | 0.98 |

## 2. Benign vs corrupting (px)

| metric | benign median | benign p90 | corrupt (>=2 px) median | separation ratio | benign NaN frac |
|---|---|---|---|---|---|
| dense_disp_px | 0.27 | 1.28 | 2.45 | 9.0x | 0.00 |
| sparse_disp_px | 0.15 | 0.53 | 2.81 | 19.2x | 0.02 |

## 3. Hallucinated structure: edge precision (mean [95% CI]; higher = fewer hallucinations)

| setting | edge_precision |
|---|---|
| n=1 | 0.916 [0.889, 0.936] |
| n=3 | 0.861 [0.842, 0.881] |
| n=6 | 0.694 [0.657, 0.730] |
| benign_edge_precision | 0.879 [0.821, 0.931] |
