"""Validate the geometry-drift metric against independent judges, on a Modal GPU, one container per image.

    modal run dgeo/drift/modal_validate.py --images data/bdd/drift_images --out results/drift_v2

For every real photo this makes
  * generated weather edits from five generator settings that keep the structure to different degrees
    (img2img with no structural control, depth ControlNet, canny ControlNet, depth+canny, and a
    neutral-prompt depth ControlNet that shows how much the generator moves things when asked for nothing),
  * synthetic edits with a KNOWN effect (benign appearance edits and geometric corruptions),
and scores every edited image with the drift metric AND with two independent judges that do not use
optical flow: semantic-segmentation agreement (SegFormer, Cityscapes) and monocular-depth rank
correlation (Depth Anything V2). Only a small CSV comes back to the laptop.
"""
from __future__ import annotations

import io
from pathlib import Path

import modal

WEATHER = {
    "fog": "a dashcam photo of the same road in thick fog, low visibility",
    "night": "a dashcam photo of the same road at night, street lights, dark sky",
    "rain": "a dashcam photo of the same road in heavy rain, wet road, grey sky",
    "snow": "a dashcam photo of the same road covered in snow, winter",
}
NEUTRAL = "a dashcam photo of a road"
NEGATIVE = "cartoon, painting, text, watermark, distorted, extra cars, deformed"
W, H_FRAC = 768, 9 / 16
SCORE_W = 480  # every pair, generated or synthetic, is scored at this width
SEEDS = (0, 1)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.1", "diffusers==0.31.0", "transformers==4.46.3", "accelerate==1.1.1", "safetensors",
                 "pillow", "numpy<2", "scipy", "pandas", "opencv-python-headless<4.11", "huggingface_hub==0.26.2")
    .add_local_python_source("dgeo")
)
cache = modal.Volume.from_name("dgeo-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("dgeo-results", create_if_missing=True)
app = modal.App("dgeo-drift-validate", image=image)


@app.cls(gpu="A10G", volumes={"/root/.cache/huggingface": cache, "/results": results}, timeout=3600,
             scaledown_window=120)
class Validator:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline,
                               UniPCMultistepScheduler)
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, pipeline

        self.torch = torch
        self.depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
        self.seg_proc = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
        self.seg = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024").to("cuda").eval()
        cn_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
        cn_canny = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16).to("cuda")
        base = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.p_depth = StableDiffusionControlNetPipeline.from_pretrained(
            base, controlnet=cn_depth, torch_dtype=torch.float16, safety_checker=None).to("cuda")
        self.p_depth.scheduler = UniPCMultistepScheduler.from_config(self.p_depth.scheduler.config)
        self.p_canny = StableDiffusionControlNetPipeline.from_pipe(self.p_depth, controlnet=cn_canny)
        self.p_multi = StableDiffusionControlNetPipeline.from_pipe(self.p_depth, controlnet=[cn_depth, cn_canny])
        self.p_i2i = StableDiffusionImg2ImgPipeline.from_pipe(self.p_depth)
        cache.commit()

    # ---- judges -------------------------------------------------------------------------------
    def _seg(self, rgb):
        import numpy as np
        t = self.seg_proc(images=rgb, return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            lg = self.seg(**t).logits
        lg = self.torch.nn.functional.interpolate(lg, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        return lg.argmax(1)[0].cpu().numpy().astype(np.uint8)

    @staticmethod
    def _iou(a, b, classes):
        ma, mb = (a[..., None] == classes).any(-1), (b[..., None] == classes).any(-1)
        u = (ma | mb).sum()
        return float((ma & mb).sum() / u) if u else float("nan")

    def _depth(self, rgb):
        import numpy as np
        from PIL import Image
        return np.array(self.depth(Image.fromarray(rgb))["depth"], dtype="float32")

    # ---- generation ---------------------------------------------------------------------------
    def _generate(self, variant, base_rgb, depth_ctrl, canny_ctrl, prompt, seed):
        torch = self.torch
        gen = torch.Generator("cuda").manual_seed(seed)
        kw = dict(negative_prompt=NEGATIVE, num_inference_steps=25, guidance_scale=7.5, generator=gen)
        if variant == "i2i":
            from PIL import Image
            return self.p_i2i(prompt, image=Image.fromarray(base_rgb), strength=0.5, **kw).images[0]
        if variant in ("depth", "noop_depth"):
            return self.p_depth(prompt, image=depth_ctrl, controlnet_conditioning_scale=1.0, **kw).images[0]
        if variant == "canny":
            return self.p_canny(prompt, image=canny_ctrl, controlnet_conditioning_scale=1.0, **kw).images[0]
        if variant == "depth_canny":
            return self.p_multi(prompt, image=[depth_ctrl, canny_ctrl],
                                controlnet_conditioning_scale=[1.0, 1.0], **kw).images[0]
        raise ValueError(variant)

    @modal.method()
    def run(self, name: str, jpg: bytes) -> str:
        import cv2
        import numpy as np
        import pandas as pd
        from PIL import Image

        from dgeo.drift import synth_edits as E
        from dgeo.drift.metrics import all_metrics

        rgb0 = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        Hh = int(round(W * H_FRAC / 8) * 8)
        base = cv2.resize(rgb0, (W, Hh), interpolation=cv2.INTER_AREA)
        d = self._depth(base)
        d = (d - d.min()) / max(float(d.max() - d.min()), 1e-6)
        depth_ctrl = Image.fromarray((np.stack([d] * 3, -1) * 255).astype("uint8"))
        canny_ctrl = Image.fromarray(np.stack([cv2.Canny(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY), 100, 200)] * 3, -1))
        seg0, dep0 = self._seg(base), self._depth(base)

        def score(edited, **meta):
            e = np.asarray(edited.convert("RGB") if hasattr(edited, "convert") else edited)
            if e.shape[:2] != base.shape[:2]:
                e = cv2.resize(e, (base.shape[1], base.shape[0]))
            s0 = cv2.resize(base, (SCORE_W, int(SCORE_W * base.shape[0] / base.shape[1])), interpolation=cv2.INTER_AREA)
            s1 = cv2.resize(e, (SCORE_W, s0.shape[0]), interpolation=cv2.INTER_AREA)
            m = all_metrics(s0, s1, False)
            sg = self._seg(e)
            from scipy.stats import spearmanr
            de = cv2.resize(self._depth(e), (dep0.shape[1], dep0.shape[0]))
            row = dict(image=name, **meta, **m,
                       seg_pixel_agreement=float((seg0 == sg).mean()),
                       seg_iou_road=self._iou(seg0, sg, np.array([0])),
                       seg_iou_vehicle=self._iou(seg0, sg, np.array([13, 14, 15])),
                       seg_iou_structure=self._iou(seg0, sg, np.array([0, 1, 2, 8, 10])),
                       depth_spearman=float(spearmanr(dep0.ravel(), de.ravel()).statistic))
            rows.append(row)

        rows: list[dict] = []
        for variant in ("noop_depth", "i2i", "depth", "canny", "depth_canny"):
            prompts = {"neutral": NEUTRAL} if variant == "noop_depth" else WEATHER
            for pname, prompt in prompts.items():
                for seed in SEEDS:
                    score(self._generate(variant, base, depth_ctrl, canny_ctrl, prompt, seed),
                          source="generated", kind=variant, setting=pname, magnitude=float("nan"), seed=seed)
        # synthetic edits with known effect, applied to the same base image
        for kind, grid in E.BENIGN.items():
            for level, kw in enumerate(grid):
                score(E.FUNCS[kind](base, **kw), source="benign", kind=kind, setting=str(kw),
                      magnitude=float(level), seed=0)
        for kind, (arg, mags) in E.CORRUPTING.items():
            for mag in mags:  # magnitudes are defined at 360 px width; scale to the 768 px base
                score(E.FUNCS[kind](base, **{arg: mag * (W / 360 if arg != "deg" else 1)}), source="corrupt",
                      kind=kind, setting=arg, magnitude=float(mag), seed=0)
        for n in (1, 3, 6):
            score(E.hallucinate(base, n), source="halluc", kind="hallucinate", setting=str(n), magnitude=float(n), seed=0)
        csv = pd.DataFrame(rows).to_csv(index=False)
        Path("/results/drift_v2").mkdir(parents=True, exist_ok=True)
        Path(f"/results/drift_v2/{name}.csv").write_text(csv)
        results.commit()
        return csv


@app.local_entrypoint()
def main(images: str = "data/bdd/drift_images", out: str = "results/drift_v2") -> None:
    import pandas as pd

    files = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    frames = []
    v = Validator()
    for f, csv in zip(files, v.run.starmap([(p.stem, p.read_bytes()) for p in files], return_exceptions=True),
                      strict=True):
        if isinstance(csv, Exception):
            print("FAILED", f.stem, type(csv).__name__, str(csv)[:200])
            continue
        frames.append(pd.read_csv(io.StringIO(csv)))
        print(f.stem, len(frames[-1]), "rows", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(Path(out) / "scores.csv", index=False)
    print("wrote", Path(out) / "scores.csv")
