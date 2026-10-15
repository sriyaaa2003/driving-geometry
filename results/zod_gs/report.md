# ZOD Gaussian Splatting and calibration sensitivity (sequence 000000, 25 held-out frames)

3D Gaussian Splatting (gsplat) trained for 12,000 steps on 176 frames of a 20 s ZOD drive, LiDAR-initialised, scored on 25 held-out frames (every 8th frame). One run per row, one seed.

| run | camera extrinsic offset | correction learned? | PSNR | SSIM | LPIPS | remaining rotation error (deg) | remaining translation error (cm) |
|---|---|---|---|---|---|---|---|
| baseline | none | no | 23.81 | 0.882 | 0.263 | 0.00 | 0.0 |
| perturbed_0.5deg_5cm_fixed | 0.5 deg, 5 cm | no | 23.46 | 0.876 | 0.283 | 0.50 | 5.0 |
| perturbed_0.5deg_5cm_learned | 0.5 deg, 5 cm | yes | 23.87 | 0.881 | 0.265 | 0.45 | 12.4 |
| perturbed_1deg_10cm_fixed | 1 deg, 10 cm | no | 22.77 | 0.868 | 0.296 | 1.00 | 10.0 |
| perturbed_1deg_10cm_learned | 1 deg, 10 cm | yes | 22.74 | 0.874 | 0.282 | 0.73 | 15.3 |
| perturbed_2deg_20cm_fixed | 2 deg, 20 cm | no | 21.34 | 0.844 | 0.348 | 2.00 | 20.0 |
| perturbed_2deg_20cm_learned | 2 deg, 20 cm | yes | 22.59 | 0.865 | 0.307 | 1.56 | 22.2 |
