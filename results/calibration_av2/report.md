# Calibrating a camera from a real drive: which directions are constrained

Argoverse 2 (val), 24 logs, 240 drive segments of 10 LiDAR sweeps (about 3 s), 720 solves. Camera A = ring_front_center (trusted), camera B = stereo_front_left or stereo_front_right, B's extrinsic is perturbed (0.5 deg and 5 cm per axis, random) and recovered from LiDAR-projected photometric consistency between the two views.

Segments with a heading change of at least 8 degrees count as curved: 26 curved, 214 straight.

Solves that ended with all three rotations within 0.1 deg of the truth: 85%.

## Error left after the solve (median over starts, then mean over segments with 95% bootstrap CI)

| axis | initial error (mean abs) | after solve, all | after solve, straight | after solve, curved | Cramer-Rao std (median) |
|---|---|---|---|---|---|
| roll about x (deg) | 0.43 | 0.03 [0.02, 0.05] | 0.03 [0.02, 0.05] | 0.02 [0.02, 0.03] | 0.004 |
| pitch about y (deg) | 0.43 | 0.08 [0.05, 0.11] | 0.08 [0.05, 0.12] | 0.05 [0.04, 0.06] | 0.003 |
| yaw about z (deg) | 0.38 | 0.05 [0.04, 0.05] | 0.05 [0.04, 0.05] | 0.06 [0.05, 0.07] | 0.003 |
| shift x, forward (cm) | 3.37 | 4.56 [4.19, 4.95] | 4.61 [4.21, 5.01] | 4.23 [3.28, 5.18] | 0.182 |
| shift y, left (cm) | 3.47 | 1.04 [0.93, 1.16] | 1.07 [0.94, 1.19] | 0.79 [0.58, 1.01] | 0.123 |
| shift z, up (cm) | 2.39 | 1.07 [0.98, 1.17] | 1.01 [0.92, 1.11] | 1.52 [1.30, 1.79] | 0.080 |

Rotations are well determined; the translation along the direction of travel (x) is the weak one. The Cramer-Rao standard deviations (last column) are much smaller than the errors actually left, so the Jacobian-based uncertainty is optimistic here.
