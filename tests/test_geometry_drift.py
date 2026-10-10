import numpy as np
import torch

from dgeo.drift import synth_edits as E
from dgeo.drift.metrics import dense_drift, sparse_drift
from dgeo.geometry import Camera, pose_error, so3_exp
from tests.fixtures import make_street_image


def test_so3_exp_is_a_rotation_and_small_angle_safe():
    w = torch.tensor([[0.3, -0.2, 0.5], [0.0, 0.0, 0.0], [1e-9, 0, 0]], dtype=torch.float64)
    R = so3_exp(w)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand(3, 3, 3), atol=1e-12)
    assert torch.allclose(R[1], torch.eye(3, dtype=torch.float64))


def test_fisheye_project_unproject_roundtrip():
    cam = Camera("equidistant", 80, 80, 79.5, 63.5, 160, 128)
    u, v = torch.tensor([30.0, 100.0, 79.5], dtype=torch.float64), torch.tensor([20.0, 90.0, 63.5], dtype=torch.float64)
    ray = cam.rays(u, v)
    uv, ok = cam.project(ray * 7.0)
    assert ok.all() and torch.allclose(uv[:, 0], u, atol=1e-6) and torch.allclose(uv[:, 1], v, atol=1e-6)


def test_pose_error_zero_for_identical_poses():
    T = np.eye(4)
    assert pose_error(T, T) == (0.0, 0.0)

def test_drift_metric_separates_benign_from_geometric():
    img = make_street_image(seed=3)
    benign = max(dense_drift(img, E.fog(img))["dense_disp_px"], dense_drift(img, E.night(img))["dense_disp_px"])
    bad = dense_drift(img, E.shift(img, 4.0))["dense_disp_px"]
    assert bad > 3 * benign and bad > 2.0
    assert sparse_drift(img, E.shift(img, 4.0))["sparse_disp_px"] > 2.0

