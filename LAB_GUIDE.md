# Lab computer guide

Goal: run the GPU and heavy parts on the lab desktop, bring back plain result files, and leave nothing behind.
You do **not** need Claude Code, GitHub login or any account on the lab computer.

## What each side does

| You (lab computer) | Me (laptop session) |
|---|---|
| Clone the repo, install, run the commands below, copy results home. | Read your results, fix whatever failed, write the real-data loaders, analyse and update the READMEs. |
| Register for KITTI-360 yourself (needs your name and purpose). | Never touch your accounts or credentials. |
| Download datasets (small subsets first). | Tell you exact files and sizes before you download anything. |

## Step by step

**0. Before you leave home:** make sure the repo is on GitHub (see the end of my reply for the link). If it is
private, the lab computer cannot clone it without logging in; either make it public (`gh repo edit --visibility public
--accept-visibility-change-consequences`) or download the ZIP from the GitHub page while logged in on your own device
and carry it on a USB stick.

**1. Open a terminal** (PowerShell on Windows, bash on Linux) and check the basics:

```bash
python --version        # need 3.10 or newer
nvidia-smi              # shows GPU model, memory and the CUDA version in the top right
git --version
```

If `nvidia-smi` is not found, tell me; the GPU parts will not run and I will replan.

**2. Get the code** into a short path (long Windows paths break things):

```bash
cd ~            # or C:\work on Windows
git clone https://github.com/sriyaaa2003/driving-geometry.git
cd driving-geometry
```

**3. Make a virtual environment** (keeps everything inside this folder, easy to delete):

```bash
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# Linux / macOS:       source .venv/bin/activate
```

**4. Install PyTorch matching the GPU.** Look at the CUDA version in `nvidia-smi` and pick one line:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4 or newer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1 to 12.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
```

**5. Install the rest:**

```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt
pip install -e . --no-deps
```

If `gsplat` fails to build, copy the last 30 lines of the error to me; do not fight it for more than 10 minutes.

**6. Check the machine** and keep the output:

```bash
python scripts/lab_check.py
```

**7. Run the CPU studies** (about 20 minutes, safe to leave running):

```bash
python scripts/lab_run.py studies
```

**8. Run the Gaussian-splatting smoke test** (GPU, first real test of `gs_calib.py`):

```bash
python scripts/lab_run.py gs-smoke
```

Two runs happen: one where camera corrections are optimised, one control with the extrinsics frozen at the truth.
If the first one reduces the translation error and the control shows a low, flat loss, the pipeline works.

**9. Real data, small first** (ZOD "mini" frames; I do not know the exact flags, so ask the tool):

```bash
pip install zod
zod download --help          # paste me this output; I will give you the exact command and the size
```

Only download after I have told you the size and you are happy with it. Then:

```bash
python -m dgeo.curation.prepare_zod inspect --root data/zod --version mini     # paste me the output
```

That output tells me which tag columns really exist; I will then fix `prepare_zod.py` and give you the next commands
(`manifest`, `embed_cli`, `run`). Embedding about 10 000 images with `facebook/dinov2-small` takes minutes on a GPU.

**10. Bring the results home:**

```bash
python scripts/pack_results.py
```

This writes `results/pack_<time>.zip` (reports, JSON and CSV only, no datasets). Email it to yourself or copy it
to a USB stick. Also send me `results/lab_check.json` and, if anything failed, `results/logs/*.log`.

**11. Leave nothing behind:** delete the folder (`rm -rf driving-geometry` or delete it in Explorer), log out of
any browser or email you opened, and do not save passwords or tokens on the machine.

## If something fails

* **Out of GPU memory in `gs_calib.py`:** run with `--init-scale 0.25` and fewer points (re-export with fewer frames
  via `--frames 4`).
* **`gsplat` fisheye error in `lab_check`:** the pinhole smoke test still tells us a lot; send the message.
* **No internet to Hugging Face:** the embedding step needs to download model weights once; ask IT whether
  `huggingface.co` is reachable, or tell me and I will switch to a model you can fetch another way.
* **Anything else:** copy the full error text, not a screenshot, and send it to me.
