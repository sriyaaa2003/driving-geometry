"""Study 6 (prep): turn a ZOD sequence into a Gaussian-Splatting scene on the Modal volume.

    modal run scripts/modal_zod_gs_prep.py::prep --seq 000000

Output (volume 'zod-data', zod_gs/<seq>/scene.npz): undistorted 962x542 front-camera images, pinhole K, camera-to-world
poses (ego pose from oxts interpolated to each image time, composed with the factory camera extrinsics), and a
colour-attributed LiDAR point cloud accumulated over the whole 20 s drive. Also writes lidar_overlay.png, which projects
the accumulated LiDAR points into one image: if poses and calibration were wrong the overlay would not sit on the scene.
"""
import modal

app = modal.App("dgeo-zod-gs-prep")
data = modal.Volume.from_name("zod-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "dataclass-wizard<1.0", "click", "zod", "numpy", "scipy", "h5py", "opencv-python-headless")
D = "/data"
W, H = 962, 542


def pose_at(em_ts, em_poses, t, Rotation, Slerp):
    import numpy as np

    i = int(np.clip(np.searchsorted(em_ts, t) - 1, 0, len(em_ts) - 2))
    a = (t - em_ts[i]) / (em_ts[i + 1] - em_ts[i])
    a = float(np.clip(a, 0.0, 1.0))
    T = np.eye(4)
    T[:3, 3] = (1 - a) * em_poses[i][:3, 3] + a * em_poses[i + 1][:3, 3]
    rots = Rotation.from_matrix([em_poses[i][:3, :3], em_poses[i + 1][:3, :3]])
    T[:3, :3] = Slerp([0, 1], rots)(a).as_matrix()
    return T


@app.function(image=image, volumes={D: data}, timeout=3600, memory=32768, cpu=8)
def prep(seq: str = "000000", voxel: float = 0.2, max_points: int = 600000):
    import os

    import cv2
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    from zod import ZodSequences
    from zod.constants import Anonymization, Camera, Lidar

    data.reload()
    root = f"{D}/zod_mini/x"
    zs = ZodSequences(dataset_root=root, version="mini")
    s = zs[seq]
    cam = s.calibration.cameras[Camera.FRONT]
    K = np.array(cam.intrinsics)[:3, :3]
    dist = np.array(cam.distortion, dtype=np.float64)
    ext = np.array(cam.extrinsics.transform)                 # camera -> vehicle reference frame
    em = s.ego_motion
    em_ts, em_poses = np.asarray(em.timestamps), np.asarray(em.poses)
    frames = s.info.get_camera_frames(Anonymization.BLUR)
    print("frames", len(frames), "ego samples", len(em_ts), "dist", dist)

    # pinhole output: 962x542, f = 400 px (about 100 degrees horizontal)
    Knew = np.array([[400.0, 0, W / 2], [0, 400.0, H / 2], [0, 0, 1]])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, dist, np.eye(3), Knew, (W, H), cv2.CV_16SC2)
    imgs, c2w, times = [], [], []
    base = pose_at(em_ts, em_poses, frames[0].time.timestamp(), Rotation, Slerp)
    base_inv = np.linalg.inv(base)
    for f in frames:
        t = f.time.timestamp()
        im = cv2.imread(str(f.filepath))
        full = cv2.remap(im, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        imgs.append(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
        # vehicle->world pose relative to the first frame, then camera->vehicle
        c2w.append(base_inv @ pose_at(em_ts, em_poses, t, Rotation, Slerp) @ ext)
        times.append(t)
    imgs, c2w, times = np.stack(imgs), np.stack(c2w), np.array(times)
    path = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1).sum()
    print(f"camera path {path:.1f} m in {times[-1] - times[0]:.1f} s")

    # accumulate LiDAR into the first-frame world
    lext = np.array(s.calibration.lidars[Lidar.VELODYNE].extrinsics.transform)          # lidar -> vehicle
    pts = []
    for lf in s.info.get_lidar_frames(lidar=Lidar.VELODYNE):
        p = np.asarray(lf.read().points, dtype=np.float64)
        T = base_inv @ pose_at(em_ts, em_poses, lf.time.timestamp(), Rotation, Slerp) @ lext
        pts.append((p @ T[:3, :3].T + T[:3, 3]).astype(np.float32))
    pts = np.concatenate(pts)
    print("raw lidar points", len(pts))
    # keep points near the camera path, then voxel-downsample
    cpos = c2w[:, :3, 3]
    keep = np.zeros(len(pts), bool)
    for c in cpos[::10]:
        keep |= np.linalg.norm(pts - c, axis=1) < 60.0
    pts = pts[keep]
    key = np.floor(pts / voxel).astype(np.int64)
    _, uniq = np.unique(key, axis=0, return_index=True)
    pts = pts[uniq]
    print("points after crop + voxel:", len(pts))

    # colours: project each point into the 5 nearest-in-time-to-closest frames, average where in view
    w2c = np.linalg.inv(c2w)
    col = np.zeros((len(pts), 3), np.float64)
    cnt = np.zeros(len(pts))
    for fi in range(0, len(frames), 4):
        pc = pts @ w2c[fi][:3, :3].T + w2c[fi][:3, 3]
        z = pc[:, 2]
        u = Knew[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + Knew[0, 2]
        v = Knew[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + Knew[1, 2]
        ok = (z > 1.0) & (z < 60) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        col[ok] += imgs[fi][v[ok].astype(int), u[ok].astype(int)]
        cnt[ok] += 1
    seen = cnt > 0
    pts, col = pts[seen], col[seen] / cnt[seen, None]
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, col = pts[sel], col[sel]
    print("coloured points:", len(pts))

    # overlay check: project accumulated LiDAR into frame 100
    fi = min(100, len(frames) - 1)
    pc = pts @ w2c[fi][:3, :3].T + w2c[fi][:3, 3]
    ok = pc[:, 2] > 1.0
    u = Knew[0, 0] * pc[ok, 0] / pc[ok, 2] + Knew[0, 2]
    v = Knew[1, 1] * pc[ok, 1] / pc[ok, 2] + Knew[1, 2]
    z = pc[ok, 2]
    ov = cv2.cvtColor(imgs[fi], cv2.COLOR_RGB2BGR).copy()
    for uu, vv, zz in zip(u, v, z):
        if 0 <= uu < W and 0 <= vv < H:
            c = cv2.applyColorMap(np.uint8([[min(255, int(zz * 6))]]), cv2.COLORMAP_JET)[0, 0].tolist()
            cv2.circle(ov, (int(uu), int(vv)), 1, c, -1)
    out = f"{D}/zod_gs/{seq}"
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(f"{out}/lidar_overlay.png", ov)
    cv2.imwrite(f"{out}/frame_100.png", cv2.cvtColor(imgs[fi], cv2.COLOR_RGB2BGR))
    np.savez_compressed(f"{out}/scene.npz", imgs=imgs, K=Knew, c2w=c2w, times=times, pts=pts, col=col.astype(np.float32),
                        ext=ext, K_orig=K, dist=dist)
    data.commit()
    print("saved", out)
