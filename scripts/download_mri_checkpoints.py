#!/usr/bin/env python3
"""Download pretrained MRI reconstruction checkpoints from official sources."""
import os
import sys
import urllib.request
import zipfile
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "reference", "mri")
os.makedirs(CKPT_DIR, exist_ok=True)

CHECKPOINTS = {
    # ---- Facebook Research VarNet (official fastMRI) ----
    "varnet_knee": {
        "url": "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/knee_leaderboard_state_dict.pt",
        "filename": "varnet_knee_leaderboard.pt",
        "source": "Facebook Research fastMRI",
        "dataset": "fastMRI knee multi-coil",
    },
    "varnet_brain": {
        "url": "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/brain_leaderboard_state_dict.pt",
        "filename": "varnet_brain_leaderboard.pt",
        "source": "Facebook Research fastMRI",
        "dataset": "fastMRI brain multi-coil",
    },
    # ---- Score-MRI (diffusion-based) ----
    "score_mri": {
        "url": "https://www.dropbox.com/s/27gtxkmh2dlkho9/checkpoint_95.pth?dl=1",
        "filename": "score_mri_checkpoint_95.pth",
        "source": "Chung & Ye, Med Image Anal 2022",
        "dataset": "fastMRI knee",
    },
    # ---- DnCNN for PnP (from KAIR) ----
    "dncnn_25": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_25.pth",
        "filename": "dncnn_25.pth",
        "source": "cszn/KAIR (Zhang et al.)",
        "dataset": "BSD68/ImageNet (sigma=25)",
    },
    "dncnn_gray_blind": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth",
        "filename": "dncnn_gray_blind.pth",
        "source": "cszn/KAIR (Zhang et al.)",
        "dataset": "BSD68/ImageNet (blind denoiser)",
    },
    # ---- DIRECT Model Zoo (KIKI-Net, VarNet from Calgary-Campinas) ----
    "kikinet_direct": {
        "url": "https://s3.aiforoncology.nl/direct-project/kikinet.zip",
        "filename": "kikinet_direct.zip",
        "source": "DIRECT Model Zoo (NKI-AI)",
        "dataset": "Calgary-Campinas brain",
        "extract": True,
    },
}

# Additional checkpoints that need special download (Google Drive / HuggingFace)
GDRIVE_CHECKPOINTS = {
    "humus_net_knee_x8": {
        "gdrive_id": "1sFXNloOn35FaV8uTk7Iy3BBvdBTZlGGf",
        "filename": "humus_net_knee_x8.zip",
        "source": "Fabian et al., NeurIPS 2022",
        "dataset": "fastMRI knee multi-coil 8x",
    },
    "mambarecon": {
        "gdrive_folder": "1aPCqYbREsk5Q-vO8aXwDLFF51pe8XPCq",
        "filename": "mambarecon_ckpts/",
        "source": "Korkmaz et al., WACV 2025",
        "dataset": "fastMRI",
    },
    "promptmr_4x_8x": {
        "gdrive_id": "1afLCO3C_S4e-q7QCt04Ksmv34jETrFQ7",
        "filename": "promptmr_4x_8x.zip",
        "source": "Bai et al., MICCAI 2023",
        "dataset": "fastMRI knee multi-coil 4x+8x",
    },
}

HF_CHECKPOINTS = {
    "promptmr_plus_knee": {
        "repo": "hellopipu/PromptMR",
        "file": "recon-fm-knee.zip",
        "filename": "promptmr_plus_knee.zip",
        "source": "Bai et al., ECCV 2024 (PromptMR+)",
        "dataset": "fastMRI knee",
    },
    "promptmr_plus_brain": {
        "repo": "hellopipu/PromptMR",
        "file": "recon-fm-brain.zip",
        "filename": "promptmr_plus_brain.zip",
        "source": "Bai et al., ECCV 2024 (PromptMR+)",
        "dataset": "fastMRI brain",
    },
}


def download_file(url, dest, desc=""):
    """Download a file with progress."""
    if os.path.exists(dest):
        print(f"  [SKIP] {os.path.basename(dest)} already exists")
        return True
    print(f"  Downloading {desc or os.path.basename(dest)}...", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 / total
                        mb = downloaded / (1024 * 1024)
                        print(f"\r    {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
            print(f"\r    Done: {downloaded / (1024*1024):.1f} MB", flush=True)
        return True
    except Exception as e:
        print(f"  [FAIL] {e}", flush=True)
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_gdrive(file_id, dest):
    """Download from Google Drive."""
    if os.path.exists(dest):
        print(f"  [SKIP] {os.path.basename(dest)} already exists")
        return True
    try:
        import gdown
        print(f"  Downloading from Google Drive...", flush=True)
        gdown.download(id=file_id, output=dest, quiet=False)
        return os.path.exists(dest)
    except ImportError:
        # Fallback: direct URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        return download_file(url, dest, "Google Drive file")


def download_hf(repo, file_path, dest):
    """Download from HuggingFace."""
    if os.path.exists(dest):
        print(f"  [SKIP] {os.path.basename(dest)} already exists")
        return True
    url = f"https://huggingface.co/{repo}/resolve/main/{file_path}"
    return download_file(url, dest, f"HuggingFace: {repo}/{file_path}")


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"  Extracting {os.path.basename(zip_path)}...", flush=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        print(f"  Extracted to {extract_to}", flush=True)
        return True
    except Exception as e:
        print(f"  [FAIL] Extract error: {e}", flush=True)
        return False


def main():
    print(f"=" * 70)
    print(f"MRI Checkpoint Downloader")
    print(f"Target: {CKPT_DIR}")
    print(f"=" * 70)

    success = 0
    failed = 0

    # Direct URL downloads
    print(f"\n--- Direct URL Downloads ---")
    for key, info in CHECKPOINTS.items():
        dest = os.path.join(CKPT_DIR, info["filename"])
        print(f"\n[{key}] {info['source']} ({info['dataset']})")
        ok = download_file(info["url"], dest, info["filename"])
        if ok and info.get("extract"):
            extract_dir = os.path.join(CKPT_DIR, key)
            os.makedirs(extract_dir, exist_ok=True)
            extract_zip(dest, extract_dir)
        if ok:
            success += 1
        else:
            failed += 1

    # Google Drive downloads
    print(f"\n--- Google Drive Downloads ---")
    for key, info in GDRIVE_CHECKPOINTS.items():
        dest = os.path.join(CKPT_DIR, info["filename"])
        print(f"\n[{key}] {info['source']} ({info['dataset']})")
        if "gdrive_id" in info:
            ok = download_gdrive(info["gdrive_id"], dest)
        else:
            print(f"  [SKIP] Folder download not supported via script. Use: ")
            print(f"    gdown --folder {info['gdrive_folder']} -O {CKPT_DIR}/{key}/")
            ok = False
        if ok:
            success += 1
        else:
            failed += 1

    # HuggingFace downloads
    print(f"\n--- HuggingFace Downloads ---")
    for key, info in HF_CHECKPOINTS.items():
        dest = os.path.join(CKPT_DIR, info["filename"])
        print(f"\n[{key}] {info['source']} ({info['dataset']})")
        ok = download_hf(info["repo"], info["file"], dest)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"DONE: {success} downloaded, {failed} failed")
    print(f"\nCheckpoints in: {CKPT_DIR}")

    # List what we have
    print(f"\nFiles:")
    for f in sorted(os.listdir(CKPT_DIR)):
        fpath = os.path.join(CKPT_DIR, f)
        if os.path.isfile(fpath):
            sz = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f:50s} {sz:8.1f} MB")
        elif os.path.isdir(fpath):
            print(f"  {f:50s} [dir]")


if __name__ == "__main__":
    main()
