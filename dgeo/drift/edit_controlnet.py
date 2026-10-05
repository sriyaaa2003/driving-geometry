"""Weather-edit driving images with a depth-conditioned ControlNet (GPU recommended).

STATUS: written but NOT yet run on a GPU.  `pip install diffusers accelerate safetensors` (see requirements-gpu.txt).

  python -m dgeo.drift.edit_controlnet --images data/imgs --out data/edited \
      --prompts "rainy night" "foggy morning" "heavy snow" --strength 0.6
Then score each output folder with  python -m dgeo.drift.score_pairs --orig data/imgs --edited data/edited/<prompt>.

Conditioning is a MONOCULAR depth map (Depth Anything V2) -- a stand-in for the occupancy / LiDAR-projected depth
a driving world model would use. If you have LiDAR-projected depth images, pass them with --depth-dir (same
file names, 8-bit grayscale, near = bright) and the monocular model is skipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", nargs="+", default=["rainy night", "foggy morning", "heavy snow"])
    ap.add_argument("--strength", type=float, default=0.6, help="img2img strength: lower keeps more of the original")
    ap.add_argument("--control-scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--size", type=int, default=512, help="long side; multiple of 8")
    ap.add_argument("--depth-dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sd", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--controlnet", default="lllyasviel/sd-controlnet-depth")
    a = ap.parse_args()

    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU; this will be extremely slow on CPU.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    cn = ControlNetModel.from_pretrained(a.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(a.sd, controlnet=cn, torch_dtype=dtype,
                                                                  safety_checker=None).to(dev)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    depth_pipe = None
    if not a.depth_dir:
        from transformers import pipeline

        depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                              device=0 if dev == "cuda" else -1)

    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if a.limit:
        paths = paths[:a.limit]
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        s = a.size / max(w, h)
        w2, h2 = int(round(w * s / 8) * 8), int(round(h * s / 8) * 8)
        img = img.resize((w2, h2))
        orig_dir = Path(a.out) / "_originals_resized"  # same-size originals so score_pairs compares like with like
        orig_dir.mkdir(parents=True, exist_ok=True)
        img.save(orig_dir / f"{p.stem}.png")
        if a.depth_dir:
            depth = Image.open(Path(a.depth_dir) / (p.stem + ".png")).convert("L").resize((w2, h2))
        else:
            d = np.array(depth_pipe(img)["depth"], dtype=np.float32)
            d = (255 * (d - d.min()) / max(d.max() - d.min(), 1e-6)).astype(np.uint8)
            depth = Image.fromarray(d).resize((w2, h2))
        control = Image.merge("RGB", [depth] * 3)
        for prompt in a.prompts:
            gen = torch.Generator(device=dev).manual_seed(a.seed)
            out = pipe(prompt=f"a photo of a road scene, {prompt}", negative_prompt="distorted, extra objects, text",
                       image=img, control_image=control, strength=a.strength, num_inference_steps=a.steps,
                       controlnet_conditioning_scale=a.control_scale, generator=gen).images[0]
            dst = Path(a.out) / prompt.replace(" ", "_")
            dst.mkdir(parents=True, exist_ok=True)
            out.save(dst / f"{p.stem}.png")
        print(f"edited {p.name}", flush=True)
    print(f"done. score with: python -m dgeo.drift.score_pairs --orig {a.out}/_originals_resized --edited {a.out}/<prompt>")


if __name__ == "__main__":
    main()
