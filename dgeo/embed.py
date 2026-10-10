"""Image embeddings from self-supervised / vision-language foundation models, with an on-disk cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def pick_device(device: str | None = None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedder(model_name: str = "facebook/dinov2-small", device: str | None = None):
    """Returns (embed_fn(list[PIL.Image]) -> np.ndarray (n,d) L2-normalised, device)."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    dev = pick_device(device)
    proc = AutoImageProcessor.from_pretrained(model_name)
    is_clip = "clip" in model_name.lower()
    if is_clip:
        from transformers import CLIPModel

        model = CLIPModel.from_pretrained(model_name).to(dev).eval()
    else:
        model = AutoModel.from_pretrained(model_name).to(dev).eval()

    @torch.no_grad()
    def embed(images) -> np.ndarray:
        inputs = proc(images=images, return_tensors="pt").to(dev)
        if is_clip:
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
        else:
            out = model(**inputs).last_hidden_state
            feats = torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=-1)  # CLS + mean patch token
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy()

    return embed, dev


def embed_paths(paths, model_name="facebook/dinov2-small", batch_size=32, device=None,
                cache: str | Path | None = None, ids=None) -> np.ndarray:
    """Embed image files in batches. If `cache` exists it is loaded; otherwise it is written (npz: emb, ids)."""
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True      # a few downloaded frames can be cut short; decode what is there

    if cache is not None and Path(cache).exists():
        return np.load(cache, allow_pickle=True)["emb"]
    embed, dev = load_embedder(model_name, device)
    out = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        out.append(embed(imgs))
        if (i // batch_size) % 20 == 0:
            print(f"[embed] {i + len(imgs)}/{len(paths)} on {dev}", flush=True)
    emb = np.concatenate(out).astype(np.float32)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, emb=emb, ids=np.array(ids if ids is not None else list(map(str, paths))),
                 model=model_name)
    return emb
