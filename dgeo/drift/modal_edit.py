"""Depth-conditioned weather edits of real dashcam images, generated on a Modal GPU.

    modal run dgeo/drift/modal_edit.py --images data/bdd/drift_images --out data/edited

Each image gets a depth map (Depth Anything V2 small), then Stable Diffusion 1.5 + ControlNet (depth)
repaints it with a weather/time prompt while the depth map holds the layout. Output goes to
<out>/<edit name>/<image name>.jpg, ready for `python -m dgeo.drift.score_pairs`.
"""
from __future__ import annotations

import io
from pathlib import Path

import modal

EDITS = {
    "fog": "a dashcam photo of the same road in thick fog, low visibility",
    "night": "a dashcam photo of the same road at night, street lights, dark sky",
    "rain": "a dashcam photo of the same road in heavy rain, wet road, grey sky",
    "snow": "a dashcam photo of the same road covered in snow, winter",
}
NEGATIVE = "cartoon, painting, text, watermark, distorted, extra cars, deformed"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "diffusers==0.31.0", "transformers==4.46.3", "accelerate==1.1.1",
                 "safetensors", "pillow", "numpy<2", "huggingface_hub==0.26.2")
)
cache = modal.Volume.from_name("dgeo-hf-cache", create_if_missing=True)
app = modal.App("dgeo-drift-edit", image=image)


@app.cls(gpu="A10G", volumes={"/root/.cache/huggingface": cache}, timeout=3600, scaledown_window=120)
class Editor:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
        from transformers import pipeline

        self.depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        cache.commit()

    @modal.method()
    def edit(self, name: str, jpg: bytes, edits: dict[str, str], seed: int = 0) -> dict[str, bytes]:
        import numpy as np
        import torch
        from PIL import Image

        im = Image.open(io.BytesIO(jpg)).convert("RGB")
        w, h = im.size
        W, H = 768, int(round(768 * h / w / 8) * 8)
        im = im.resize((W, H))
        d = self.depth(im)["depth"].resize((W, H))
        d = np.asarray(d, dtype=np.float32)
        d = (d - d.min()) / max(float(d.max() - d.min()), 1e-6)
        ctrl = Image.fromarray((np.stack([d] * 3, -1) * 255).astype("uint8"))
        out = {}
        for key, prompt in edits.items():
            gen = torch.Generator("cuda").manual_seed(seed)
            res = self.pipe(prompt, image=ctrl, negative_prompt=NEGATIVE, num_inference_steps=25,
                            guidance_scale=7.5, controlnet_conditioning_scale=1.0, generator=gen).images[0]
            buf = io.BytesIO()
            res.save(buf, format="JPEG", quality=95)
            out[key] = buf.getvalue()
        return out


@app.local_entrypoint()
def main(images: str = "data/bdd/drift_images", out: str = "data/edited", seed: int = 0) -> None:
    files = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    editor = Editor()
    results = editor.edit.starmap([(p.stem, p.read_bytes(), EDITS, seed) for p in files])
    for p, res in zip(files, results, strict=True):
        for key, data in res.items():
            d = Path(out) / key
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{p.stem}.jpg").write_bytes(data)
    print(f"wrote {len(files)} images x {len(EDITS)} edits to {out}")
