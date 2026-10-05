import numpy as np
import torch

from dgeo.calibration.core import CalibProblem, observability
from dgeo.calibration.scene import make_scene
from dgeo.drift import synth_edits as E
from dgeo.drift.metrics import dense_drift, sparse_drift
from dgeo.geometry import Camera, pose_error, so3_exp
from dgeo.synth import make_street_image


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


def test_residual_grows_with_rotation_error():
    s = make_scene("straight", n_points=900, size=(96, 120), focal=41.0, n_frames=3)
    pb = CalibProblem(s, s.R_cl, s.t_cl)
    p0 = torch.zeros(18, dtype=torch.float64)
    m = pb.make_mask(p0, 0)
    r0 = float((pb.residual(p0, 0, m) ** 2).mean())
    p = p0.clone()
    p[0::6] = 3.0  # 3 deg about x for every camera
    assert float((pb.residual(p, 0, m) ** 2).mean()) > 2 * r0


def test_translation_along_surfaces_is_weaker_than_normal_to_ground():
    """A shift of all cameras along z (ground normal) is better observed than one along x (tangent to the
    ground and both walls) on a straight path -- the mechanism behind the degeneracy study."""
    s = make_scene("straight", n_points=1500, size=(96, 120), focal=41.0, n_frames=4)
    o = observability(CalibProblem(s, s.R_cl, s.t_cl, sigma_c=0.05))
    c = o["common_mode_std_cm"]
    assert c["z"] < c["x"]


def test_pose_error_zero_for_identical_poses():
    T = np.eye(4)
    assert pose_error(T, T) == (0.0, 0.0)


def test_drift_metric_separates_benign_from_geometric():
    img = make_street_image(seed=3)
    benign = max(dense_drift(img, E.fog(img))["dense_disp_px"], dense_drift(img, E.night(img))["dense_disp_px"])
    bad = dense_drift(img, E.shift(img, 4.0))["dense_disp_px"]
    assert bad > 3 * benign and bad > 2.0
    assert sparse_drift(img, E.shift(img, 4.0))["sparse_disp_px"] > 2.0
