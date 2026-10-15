"""Study 5, step 1: embed the 100,000 Zenseact Open Dataset frames on a Modal GPU. The benchmark itself is dgeo/zod/run_curation.py.

    modal run scripts/modal_zod_curation.py::embed            # GPU: DINOv2-small + CLIP ViT-B/32 embeddings of all frames

Inputs on the 'zod-data' volume (built earlier by the zod-fssl repo): shards/shard_XXXX.npy (uint8, N x 3 x 144 x 256)
and index.parquet (country, time, day/night, road condition, road type, ...). Nothing is downloaded locally.
"""
import modal

app = modal.App("dgeo-zod-curation")
data = modal.Volume.from_name("zod-data")
common = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]
gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install(*common, "torch", "transformers==4.46.3", "pillow")
             .add_local_python_source("dgeo"))
D = "/data"
OUT = f"{D}/zod_curation"


@app.function(image=gpu_image, gpu="A10G", volumes={D: data}, timeout=3 * 3600)
def embed():
    import json
    import os

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, CLIPModel

    data.reload()
    os.makedirs(OUT, exist_ok=True)
    man = json.load(open(f"{D}/shards/manifest.json"))
    dev = "cuda"
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval().half()
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval().half()

    def stat(v):
        return torch.tensor(v, device=dev).view(1, 3, 1, 1)

    im_mean, im_std = stat([0.485, 0.456, 0.406]), stat([0.229, 0.224, 0.225])
    c_mean, c_std = stat([0.4815, 0.4578, 0.4082]), stat([0.2686, 0.2613, 0.2758])
    e_dino, e_clip = [], []
    for f in man["files"]:
        arr = np.load(f"{D}/shards/{f}")                              # N,3,144,256 uint8
        for i in range(0, len(arr), 250):
            x = torch.from_numpy(arr[i:i + 250]).to(dev).float() / 255.0
            with torch.no_grad():
                xd = F.interpolate(x, size=(140, 252), mode="bilinear", align_corners=False)   # multiples of 14, keeps aspect
                o = dino(pixel_values=((xd - im_mean) / im_std).half()).last_hidden_state
                e_dino.append(F.normalize(torch.cat([o[:, 0], o[:, 1:].mean(1)], -1).float(), dim=-1).cpu().numpy())
                xc = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                fc = clip.get_image_features(pixel_values=((xc - c_mean) / c_std).half())
                e_clip.append(F.normalize(fc.float(), dim=-1).cpu().numpy())
        print("embedded", f, flush=True)
    ids = np.array([f"{i:06d}" for i in range(man["n"])])
    for name, e in (("dinov2-small", e_dino), ("clip-vit-base-patch32", e_clip)):
        e = np.concatenate(e).astype(np.float32)
        assert len(e) == man["n"] and np.isfinite(e).all()
        np.savez(f"{OUT}/emb_{name}.npz", emb=e, ids=ids, model=name)
        print(name, e.shape)
    data.commit()
