# Calibration study (synthetic fisheye rig; CPU)

Photometric multi-view consistency of LiDAR points with known odometry. **Synthetic data**: these are properties of the method on a controlled scene, not results on a real vehicle.

## 1. Observability at the true calibration

`common-mode std` = marginal std (cm) of shifting ALL cameras together along a LiDAR-frame axis. The linear (Cramer-Rao) numbers are optimistic: they ignore systematic model error, which is why the empirical errors in section 2 are larger.

| scene | prior | lambda_min | common-mode std x / y / z (cm) | max rot std (deg) | max trans std (cm) |
|---|---|---|---|---|---|
| straight+walls | none | 0.40 | 1.53 / 1.45 / 0.81 | 0.084 | 1.39 |
| straight+walls | road-plane | 0.41 | 1.53 / 1.45 / 0.75 | 0.078 | 1.39 |
| curve+walls | none | 0.38 | 1.52 / 1.33 / 0.84 | 0.095 | 1.46 |
| curve+walls | road-plane | 0.38 | 1.51 / 1.33 / 0.77 | 0.087 | 1.46 |
| straight+open | none | 0.10 | 2.22 / 3.05 / 0.68 | 0.096 | 2.47 |
| straight+open | road-plane | 0.11 | 2.21 / 3.04 / 0.64 | 0.089 | 2.47 |
| curve+open | none | 0.09 | 2.16 / 3.04 / 0.62 | 0.123 | 2.94 |
| curve+open | road-plane | 0.10 | 2.15 / 3.02 / 0.58 | 0.123 | 2.91 |

## 2. Empirical error of a coarse-to-fine solve from a perturbed start

Mean over cameras and trials. Position error is in the LiDAR frame.

| scene | prior | start | rot err (deg) | trans err (cm) | pos err x / y / z (cm) |
|---|---|---|---|---|---|
| curve+open | none | 1.0 deg / 3.0 cm | 0.24 | 14.6 | 13.7 / 3.7 / 0.8 |
| curve+open | road-plane | 1.0 deg / 3.0 cm | 0.24 | 14.2 | 13.0 / 4.7 / 0.8 |
| curve+walls | none | 1.0 deg / 3.0 cm | 0.08 | 4.4 | 4.2 / 0.2 / 0.6 |
| curve+walls | road-plane | 1.0 deg / 3.0 cm | 0.08 | 4.4 | 4.3 / 0.2 / 0.4 |
| straight+open | none | 1.0 deg / 3.0 cm | 0.34 | 26.0 | 4.0 / 25.3 / 0.4 |
| straight+open | road-plane | 1.0 deg / 3.0 cm | 0.33 | 24.5 | 3.2 / 23.8 / 0.3 |
| straight+walls | none | 1.0 deg / 3.0 cm | 0.12 | 8.8 | 8.7 / 1.6 / 0.3 |
| straight+walls | road-plane | 1.0 deg / 3.0 cm | 0.11 | 8.5 | 8.4 / 1.3 / 0.2 |
