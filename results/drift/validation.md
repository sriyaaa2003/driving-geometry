# Geometry-drift metric validation

Images with KNOWN edits. A trustworthy metric must (1) rise with injected geometric corruption, (2) stay low for benign appearance edits, (3) notice hallucinated structure.

Images: 30, edits per image: 30.

## 1. Dose-response (Spearman rho between injected magnitude and metric; 1.0 = perfect)

| corruption : metric | rho |
|---|---|
| elastic:dense_disp_p90 | 0.98 |
| elastic:dense_disp_px | 0.98 |
| elastic:rel_rot_deg | 0.53 |
| elastic:sparse_disp_px | 0.98 |
| rotate:dense_disp_p90 | 0.98 |
| rotate:dense_disp_px | 0.98 |
| rotate:rel_rot_deg | 0.82 |
| rotate:sparse_disp_px | 0.98 |
| shift:dense_disp_p90 | 0.98 |
| shift:dense_disp_px | 0.98 |
| shift:rel_rot_deg | 0.36 |
| shift:sparse_disp_px | 0.98 |

## 2. Benign vs corrupting (px)

| metric | benign median | benign p90 | corrupt (>=2 px) median | separation ratio | benign NaN frac |
|---|---|---|---|---|---|
| dense_disp_px | 0.09 | 0.29 | 2.63 | 28.0x | 0.00 |
| sparse_disp_px | 0.26 | 0.66 | 2.56 | 9.7x | 0.00 |

## 3. Hallucinated structure: edge precision (mean [95% CI]; higher = fewer hallucinations)

| setting | edge_precision |
|---|---|
| n=1 | 0.966 [0.956, 0.975] |
| n=3 | 0.947 [0.933, 0.959] |
| n=6 | 0.890 [0.867, 0.912] |
| benign_edge_precision | 0.950 [0.939, 0.959] |
