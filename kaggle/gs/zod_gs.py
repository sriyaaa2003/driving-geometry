"""Study 6 on Kaggle: 3D Gaussian Splatting on a real ZOD drive, and camera-to-vehicle calibration through the splat.

Input (private Kaggle dataset zod-gs-scene): scene.npz built from ZOD sequence 000000 (front camera undistorted to 962x542,
oxts ego poses composed with the factory camera extrinsics, a colour-attributed LiDAR point cloud). Needs a GPU and internet
(gsplat compiles its CUDA kernels on first use). Writes metrics.json and test renders for each run to /kaggle/working/runs.

Calibration experiment: the front-camera extrinsic (camera -> vehicle) is perturbed by a known rotation and translation, then
either left wrong or optimised jointly with the Gaussians (one 6-DoF correction shared by all frames). The ZOD factory
calibration is the reference; the error reported is the remaining rotation (degrees) and translation (cm).
"""
import glob
import json
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gsplat", "lpips", "ninja", "jaxtyping", "rich"], check=False)
import torch

cc = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}.{cc[1]}"
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0), flush=True)
SCENE = glob.glob("/kaggle/input/**/scene.npz", recursive=True)[0]
D = "/kaggle/working"
HOOD_ROW = 425            # rows >= this (of 542) show the bonnet / bumper
TEST_EVERY, TEST_OFF = 8, 4


def rodrigues(rv):
    import torch

    th = torch.linalg.norm(rv) + 1e-12
    k = rv / th
    Kx = torch.zeros(3, 3, device=rv.device, dtype=rv.dtype)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=rv.device, dtype=rv.dtype) + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def se3(rv, t):
    import torch

    T = torch.eye(4, device=rv.device, dtype=rv.dtype)
    T = T.clone()
    T[:3, :3] = rodrigues(rv)
    T[:3, 3] = t
    return T


def rot_angle_deg(R):
    import numpy as np

    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def train_one(seq: str = "000000", name: str = "baseline", steps: int = 15000, rot_deg: float = 0.0,
              trans_cm: float = 0.0, learn: bool = False, seed: int = 0, ext_start: int = 1000,
              n_sky: int = 20000, save_renders: bool = True):
    import json
    import os
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from scipy.spatial import cKDTree
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

    t0 = time.time()
    dev = "cuda"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    z = np.load(SCENE)
    imgs = torch.from_numpy(z["imgs"]).to(dev)                               # N,H,W,3 uint8
    n_img, H, W, _ = imgs.shape
    K = torch.from_numpy(z["K"]).float().to(dev)
    c2w = torch.from_numpy(z["c2w"]).float().to(dev)
    pts, col = z["pts"], z["col"]
    mask = torch.zeros(1, H, W, 1, device=dev)
    mask[:, :HOOD_ROW] = 1.0
    test_idx = np.arange(TEST_OFF, n_img, TEST_EVERY)
    train_idx = np.setdiff1d(np.arange(n_img), test_idx)

    # --- Gaussians: LiDAR points (+ distant sky Gaussians) -------------------------------------------------
    c0 = c2w[:, :3, 3].mean(0).cpu().numpy()
    sky_dir = rng.normal(size=(n_sky, 3))
    sky_dir[:, 2] = np.abs(sky_dir[:, 2]) * 0.6 + 0.02
    sky_dir /= np.linalg.norm(sky_dir, axis=1, keepdims=True)
    sky_pts = c0 + 1500.0 * sky_dir
    top = z["imgs"][::10, : int(0.25 * H)].reshape(-1, 3).mean(0) / 255.0
    sky_col = np.tile(top, (n_sky, 1)) + rng.normal(scale=0.02, size=(n_sky, 3))
    all_pts = np.concatenate([pts, sky_pts]).astype(np.float32)
    all_col = np.clip(np.concatenate([col / 255.0, sky_col]), 0, 1).astype(np.float32)
    d, _ = cKDTree(pts).query(pts, k=4)
    sc = np.clip(d[:, 1:].mean(1), 0.03, 1.0)
    sc = np.concatenate([sc, np.full(n_sky, 40.0)])
    N = len(all_pts)
    C0 = 0.28209479177387814
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(all_pts).to(dev)),
        "scales": torch.nn.Parameter(torch.log(torch.from_numpy(sc).float()).unsqueeze(1).repeat(1, 3).to(dev)),
        "quats": torch.nn.Parameter(F.normalize(torch.randn(N, 4, device=dev), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.15, device=dev))),
        "sh0": torch.nn.Parameter(((torch.from_numpy(all_col).to(dev) - 0.5) / C0).unsqueeze(1)),
        "shN": torch.nn.Parameter(torch.zeros(N, 15, 3, device=dev)),
    })
    scene_scale = 10.0
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(steps * 0.65), refine_every=100,
                               reset_every=3000, prune_opa=0.005, grow_grad2d=0.0002, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    # --- calibration perturbation and (optional) learnable correction --------------------------------------
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    dr = rng.normal(size=3)
    dr /= np.linalg.norm(dr)
    P = se3(torch.tensor(ax * np.radians(rot_deg), dtype=torch.float32, device=dev),
            torch.tensor(dr * trans_cm / 100.0, dtype=torch.float32, device=dev))
    corr_rv = torch.nn.Parameter(torch.zeros(3, device=dev))
    corr_t = torch.nn.Parameter(torch.zeros(3, device=dev))
    ext_opt = torch.optim.Adam([{"params": [corr_rv], "lr": 3e-4}, {"params": [corr_t], "lr": 3e-4}])

    def cam_to_world(i):
        return c2w[i] @ P @ se3(corr_rv, corr_t)

    def render(i_list):
        view = torch.stack([torch.linalg.inv(cam_to_world(i)) for i in i_list])
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        out, alpha, info = rasterization(params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
                                         torch.sigmoid(params["opacities"]), colors, view, K[None].repeat(len(i_list), 1, 1),
                                         W, H, sh_degree=3, packed=False, near_plane=0.2, far_plane=3000.0)
        return out.clamp(0, 1), info

    def gt(i_list):
        return imgs[i_list].float() / 255.0

    t_train = time.time()
    for step in range(steps):
        bi = [int(rng.choice(train_idx))]
        pred, info = render(bi)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        target = gt(bi)
        l1 = (torch.abs(pred - target) * mask).sum() / (mask.sum() * 3 * len(bi))
        s = ssim_fn((pred * mask).permute(0, 3, 1, 2), (target * mask).permute(0, 3, 1, 2), data_range=1.0)
        loss = 0.8 * l1 + 0.2 * (1 - s)
        loss.backward()
        for k, opt in optimizers.items():
            if k == "means":
                for g in opt.param_groups:
                    g["lr"] = lrs["means"] * (0.01 ** (step / steps))
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if learn and step >= ext_start:
            ext_opt.step()
        ext_opt.zero_grad(set_to_none=True)
        if step % 1000 == 0:
            with torch.no_grad():
                Pc = P @ se3(corr_rv, corr_t)
                err_r, err_t = rot_angle_deg(Pc[:3, :3].cpu().numpy()), float(Pc[:3, 3].norm() * 100)
            print(f"[{name}] step {step} loss {loss.item():.4f} n_gauss {len(params['means'])} calib err {err_r:.3f} deg {err_t:.2f} cm "
                  f"({time.time() - t_train:.0f}s)", flush=True)

    # --- evaluation on held-out frames -------------------------------------------------------------------------
    import lpips

    lp = lpips.LPIPS(net="alex").to(dev)
    psnrs, ssims, lps, renders = [], [], [], []
    with torch.no_grad():
        for i in test_idx:
            pred, _ = render([int(i)])
            tgt = gt([int(i)])
            m = mask
            mse = (((pred - tgt) ** 2) * m).sum() / (m.sum() * 3)
            psnrs.append(float(-10 * torch.log10(mse)))
            ssims.append(float(ssim_fn((pred * m).permute(0, 3, 1, 2), (tgt * m).permute(0, 3, 1, 2), data_range=1.0)))
            lps.append(float(lp((pred * m).permute(0, 3, 1, 2) * 2 - 1, (tgt * m).permute(0, 3, 1, 2) * 2 - 1)))
            renders.append((pred[0].cpu().numpy() * 255).astype(np.uint8))
        Pc = (P @ se3(corr_rv, corr_t)).cpu().numpy()
    res = dict(name=name, seq=seq, steps=steps, rot_deg=rot_deg, trans_cm=trans_cm, learn=learn, seed=seed,
               n_gaussians=int(len(params["means"])), n_train=len(train_idx), n_test=len(test_idx),
               psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)), lpips=float(np.mean(lps)),
               psnr_per_frame=psnrs, calib_err_deg=rot_deg_err(Pc), calib_err_cm=float(np.linalg.norm(Pc[:3, 3]) * 100),
               train_seconds=time.time() - t_train)
    out = f"/kaggle/working/runs/{name}"
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/metrics.json", "w"), indent=1)
    if save_renders:
        import cv2

        for j in (2, 12, 22):
            if j < len(test_idx):
                both = np.concatenate([z["imgs"][test_idx[j]], renders[j]], axis=1)
                cv2.imwrite(f"{out}/test_{j:02d}_gt_vs_render.jpg", cv2.cvtColor(both, cv2.COLOR_RGB2BGR))
    print(f"[{name}] PSNR {res['psnr']:.2f} SSIM {res['ssim']:.3f} LPIPS {res['lpips']:.3f} calib err "
          f"{res['calib_err_deg']:.3f} deg {res['calib_err_cm']:.2f} cm ({time.time() - t0:.0f}s total)", flush=True)
    return {k: v for k, v in res.items() if k != "psnr_per_frame"}


def rot_deg_err(Pc):
    return rot_angle_deg(Pc[:3, :3])


if __name__ == "__main__":
    STEPS = int(os.environ.get("STEPS", 12000))
    print("smoke test (300 steps)", train_one(name="smoke", steps=300, save_renders=False), flush=True)   # fail fast
    jobs = [("baseline", 0.0, 0.0, False, 0)]
    for rot, cm in ((0.5, 5.0), (1.0, 10.0), (2.0, 20.0)):
        tag = f"{rot:g}deg_{cm:g}cm"
        jobs.append((f"perturbed_{tag}_fixed", rot, cm, False, 1))
        jobs.append((f"perturbed_{tag}_learned", rot, cm, True, 1))
    summary = []
    for name, rot, cm, learn, seed in jobs:
        try:
            r = train_one(name=name, steps=STEPS, rot_deg=rot, trans_cm=cm, learn=learn, seed=seed)
            summary.append(r)
        except Exception as e:  # keep going: one failed run must not lose the others
            import traceback
            traceback.print_exc()
            summary.append(dict(name=name, error=repr(e)))
        json.dump(summary, open("/kaggle/working/summary.json", "w"), indent=1)
