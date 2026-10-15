"""Extract sequences_mini on the Modal volume and print what the ZOD devkit exposes (poses, calibration, LiDAR).

    modal run scripts/modal_zod_explore.py::explore
"""
import modal

app = modal.App("dgeo-zod-explore")
data = modal.Volume.from_name("zod-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("dataclass-wizard<1.0", "click", "zod", "numpy", "h5py", "opencv-python-headless")
D = "/data"


@app.function(image=image, volumes={D: data}, timeout=3600, memory=16384)
def explore():
    import os
    import subprocess

    data.reload()
    root = f"{D}/zod_mini/x"
    if not os.path.exists(f"{root}/trainval-sequences-mini.json"):
        os.makedirs(root, exist_ok=True)
        subprocess.run(["tar", "xzf", f"{D}/zod_mini/sequences_mini.tar.gz", "-C", root], check=True)
        data.commit()
    print(subprocess.run(["bash", "-c", f"ls {root}; ls {root}/sequences; ls {root}/sequences/000000; du -sh {root}"], capture_output=True, text=True).stdout)
    print(open(f"{root}/trainval-sequences-mini.json").read()[:600])
    print(open(f"{root}/sequences/000000/calibration.json").read()[:3000])

    from zod import ZodSequences
    from zod.constants import Anonymization, Camera, Lidar

    zs = ZodSequences(dataset_root=root, version="mini")
    ids = sorted(zs.get_all_ids())
    print("ids", ids)
    s = zs[ids[0]]
    print([a for a in dir(s) if not a.startswith("_")])
    frames = s.info.get_camera_frames(Anonymization.BLUR)
    print("n camera frames", len(frames), frames[0].time, frames[1].time, frames[0].filepath)
    cal = s.calibration
    print([a for a in dir(cal) if not a.startswith("_")])
    cam = cal.cameras[Camera.FRONT]
    print("intrinsics", cam.intrinsics, "distortion", cam.distortion, "undistortion", getattr(cam, "undistortion", None), "extr", cam.extrinsics.transform, cam.field_of_view)
    em = s.ego_motion
    print([a for a in dir(em) if not a.startswith("_")])
    print("ego poses", em.poses.shape, em.timestamps[:3], em.timestamps[-1])
    lid = s.info.get_lidar_frames(lidar=Lidar.VELODYNE)
    print("n lidar frames", len(lid), lid[0].filepath)
    pc = lid[0].read()
    print([a for a in dir(pc) if not a.startswith("_")])
    print(pc.points.shape, pc.intensity.shape)
    import cv2
    im = cv2.imread(str(frames[0].filepath))
    print("image", im.shape)
