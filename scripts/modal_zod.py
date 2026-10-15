"""Download a ZOD subset into the Modal volume 'zod-data' (uses the 'zod-download' Modal secret with ZOD_URL).

    modal run scripts/modal_zod.py::download --subset sequences --version mini --dry-run
    modal run scripts/modal_zod.py::download --subset sequences --version mini
    modal run scripts/modal_zod.py::ls --path /data
"""
import modal

app = modal.App("dgeo-zod-download")
image = modal.Image.debian_slim(python_version="3.11").pip_install("dataclass-wizard<1.0", "click", "zod[cli]")
data = modal.Volume.from_name("zod-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": data}, secrets=[modal.Secret.from_name("zod-download")], timeout=24 * 3600)
def download(subset: str = "sequences", version: str = "mini", dry_run: bool = False) -> None:
    import os
    import subprocess

    cmd = ["zod", "download", "-y", f"--url={os.environ['ZOD_URL']}", "--output-dir=/data",
           f"--subset={subset}", f"--version={version}", "--images", "--blur", "--no-dnat", "--annotations",
           "--infos", "--lidar", "--oxts", "--vehicle-data"]
    if dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)
    if not dry_run:
        data.commit()


@app.function(image=image, volumes={"/data": data})
def ls(path: str = "/data", depth: int = 2) -> None:
    import subprocess

    print(subprocess.run(["bash", "-c", f"du -sh {path}; find {path} -maxdepth {depth} | head -60"],
                         capture_output=True, text=True).stdout)

