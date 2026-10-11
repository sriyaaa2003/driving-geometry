# Calibration v2: Argoverse 2, camera B recovered from camera A + LiDAR

57 logs (286 segments of 10 sweeps, about 3 s), half of them the most-turning logs of the validation split. Camera A = ring_front_center, camera B = stereo_front_left or stereo_front_right.

## 0. How exact is the reference calibration?

Across the 141 validation logs the extrinsics of camera B relative to camera A are not identical: 80 distinct values for stereo_front_left. Standard deviation across logs:

| camera B | rotation (deg, per axis) | shift (cm, per axis) |
|---|---|---|
| stereo_front_left | 0.21, 0.18, 0.28 | 0.42, 0.44, 0.44 |
| stereo_front_right | 0.25, 0.21, 0.17 | 0.42, 0.37, 0.42 |

The reference is calibrated per log, so these differences may be real re-calibrations. Section 1 shows the solver lands within 0.02-0.05 deg of each log's own reference, much closer than the 0.2 deg spread between logs. The absolute accuracy of the reference itself cannot be checked with this data.

## 1. Starting at the reference (no perturbation): does the solver stay there?

Median absolute drift away from the reference. A non-zero value is bias of the photometric model or noise in the reference, not an optimisation failure.

| axis | median abs deviation | mean [95% CI] over segments |
|---|---|---|
| roll (deg) | 0.017 | 0.022 [0.020, 0.026] |
| pitch (deg) | 0.034 | 0.038 [0.036, 0.041] |
| yaw (deg) | 0.037 | 0.045 [0.042, 0.048] |
| shift forward x (cm) | 4.460 | 4.708 [4.441, 4.977] |
| shift left y (cm) | 0.873 | 1.031 [0.963, 1.094] |
| shift up z (cm) | 1.063 | 1.139 [1.067, 1.218] |

## 2. Recovering a random perturbation (0.5 deg and 5 cm per axis)

| axis | initial error | after solve, mean [95% CI] | Cramer-Rao std | empirical std of the error across segments |
|---|---|---|---|---|
| roll (deg) | 0.43 | 0.032 [0.026, 0.040] | 0.004 | 0.209 |
| pitch (deg) | 0.41 | 0.059 [0.047, 0.074] | 0.003 | 0.465 |
| yaw (deg) | 0.39 | 0.045 [0.042, 0.048] | 0.003 | 0.133 |
| shift forward x (cm) | 3.29 | 4.653 [4.388, 4.918] | 0.187 | 4.097 |
| shift left y (cm) | 3.14 | 1.062 [0.990, 1.127] | 0.122 | 1.993 |
| shift up z (cm) | 1.89 | 1.188 [1.103, 1.280] | 0.085 | 2.228 |

Starts that end with all rotations within 0.1 deg: 83%.

## 3. How far from the truth can the solver start?

| start (deg, cm per axis) | rotations within 0.1 deg | median abs err: roll / pitch / yaw (deg) | forward / left / up (cm) |
|---|---|---|---|
| 0.25 / 2.5 | 90% | 0.017 / 0.034 / 0.037 | 4.44 / 0.86 / 1.07 |
| 0.5 / 5 | 83% | 0.019 / 0.037 / 0.038 | 4.57 / 0.92 / 1.09 |
| 1 / 10 | 31% | 0.110 / 0.148 / 0.143 | 5.78 / 2.86 / 1.79 |
| 2 / 20 | 9% | 1.267 / 1.954 / 2.028 | 9.30 / 10.54 / 6.11 |

## 4. Does the depth of the LiDAR points matter?

| points at | n segments | roll (deg) | pitch (deg) | yaw (deg) | shift forward x (cm) | shift left y (cm) | shift up z (cm) |
|---|---|---|---|---|---|---|---|
| near (4-15 m) | 350 | 0.069 | 0.064 | 0.084 | 2.662 | 1.442 | 1.182 |
| mid (15-30 m) | 553 | 0.024 | 0.050 | 0.060 | 5.289 | 1.670 | 1.039 |
| far (30-50 m) | 554 | 0.027 | 0.063 | 0.049 | 8.809 | 2.576 | 1.656 |

## 5. Does turning help? Error by how much the vehicle turned in the segment

| heading change | n segments | roll (deg) | pitch (deg) | yaw (deg) | shift forward x (cm) | shift left y (cm) | shift up z (cm) |
|---|---|---|---|---|---|---|---|
| 0-3 deg | 332 | 0.017 | 0.035 | 0.034 | 4.691 | 0.903 | 1.010 |
| 3-10 deg | 104 | 0.021 | 0.033 | 0.041 | 4.594 | 0.958 | 0.966 |
| 10-25 deg | 38 | 0.026 | 0.037 | 0.034 | 5.058 | 0.962 | 1.369 |
| 25-+ deg | 98 | 0.018 | 0.040 | 0.042 | 3.178 | 0.827 | 1.172 |

Spearman between heading change and forward-shift error: rho = -0.13 (p = 0.002, n = 572 segments).

## 6. Against a classical baseline (SIFT matches + PnP with LiDAR depth, no starting guess)

PnP solved 568 of 572 segments. On the segments both methods solved:

| axis | photometric (this work) | PnP baseline |
|---|---|---|
| roll (deg) | 0.018 | 0.049 |
| pitch (deg) | 0.036 | 0.047 |
| yaw (deg) | 0.037 | 0.055 |
| shift forward x (cm) | 4.493 | 4.109 |
| shift left y (cm) | 0.895 | 1.552 |
| shift up z (cm) | 1.078 | 1.237 |

## 7. The forward offset: a timing error or a constant?

When started at the reference the solver still moves camera B forward by a median 4.4 cm. A camera-LiDAR timing error would grow with vehicle speed. Median signed forward offset by speed:

| speed (m/s) | n segments | median forward offset (cm) |
|---|---|---|
| 0-2 | 130 | 4.23 |
| 2-5 | 174 | 3.77 |
| 5-10 | 196 | 4.65 |
| 10-+ | 72 | 5.58 |

Spearman between speed and offset: 0.10. The PnP baseline, which uses no photometric model, finds a median forward offset of 3.5 cm against the same reference. The offset is already 4 cm at standstill and grows only about 0.1 cm per m/s (a timing error of about 1 ms), far too little to explain it, and two different methods agree on it. So it is mostly a constant offset and not specific to the photometric solver. It comes from the reference calibration or from something both methods share (for example the lens model or the LiDAR frame); this data cannot tell which.

