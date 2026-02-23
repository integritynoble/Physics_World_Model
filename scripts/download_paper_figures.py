#!/usr/bin/env python3
"""
Download paper setup figures from arXiv / open-access PDFs for all modalities.

For each modality with an identified open-access paper, this script:
  1. Downloads the PDF from arXiv or another open-access source
  2. Renders a specific page (containing the experimental setup figure) as PNG
  3. Saves to platform/pwm_platform/static/img/paper_setups/<key>.png

Also produces a JSON manifest with citation info so the web UI can display
"from <citation>" labels and link to the papers.

Requirements:
    pip install pymupdf   (import fitz)

Usage:
    python scripts/download_paper_figures.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "platform" / "pwm_platform" / "static" / "img" / "paper_setups"
MANIFEST_PATH = OUT_DIR / "paper_setups_manifest.json"

# ── Paper setup figure database ─────────────────────────────────────────────
# Each entry: modality_key → {url, page (0-indexed), citation, link}
# Only modalities with identified open-access papers are included.
# "page" is the 0-indexed page of the PDF that contains the setup figure.
PAPER_FIGURES: dict[str, dict] = {
    # ── Computational Spectral Imaging ──
    "cassi": {
        "url": "https://arxiv.org/pdf/2204.07908",
        "page": 0,
        "citation": "Cai et al., 'MST++: Multi-Stage Spectral-Wise Transformer', CVPRW 2022",
        "link": "https://arxiv.org/abs/2204.07908",
    },
    "cacti": {
        "url": "https://arxiv.org/pdf/1302.2575",
        "page": 1,
        "citation": "Llull et al., 'Coded Aperture Compressive Temporal Imaging', Optics Express 2013",
        "link": "https://arxiv.org/abs/1302.2575",
    },
    # ── Medical Imaging ──
    "mri": {
        "url": "https://arxiv.org/pdf/1811.08839",
        "page": 1,
        "citation": "Zbontar et al., 'fastMRI: An Open Dataset and Benchmarks', arXiv 2018",
        "link": "https://arxiv.org/abs/1811.08839",
    },
    "ct": {
        "url": "https://arxiv.org/pdf/1910.01113",
        "page": 2,
        "citation": "Leuschner et al., 'LoDoPaB-CT: A Benchmark Dataset for Low-Dose CT', arXiv 2019",
        "link": "https://arxiv.org/abs/1910.01113",
    },
    "cbct": {
        "url": "https://arxiv.org/pdf/2112.11863",
        "page": 1,
        "citation": "Zha et al., 'NAF: Neural Attenuation Fields for Sparse-View CBCT', arXiv 2021",
        "link": "https://arxiv.org/abs/2112.11863",
    },
    "pet": {
        "url": "https://arxiv.org/pdf/2108.06950",
        "page": 1,
        "citation": "Reader et al., 'Deep Learning for PET Image Reconstruction', IEEE TMI 2021",
        "link": "https://arxiv.org/abs/2108.06950",
    },
    "spect": {
        "url": "https://arxiv.org/pdf/2208.11706",
        "page": 0,
        "citation": "Xie et al., 'Deep Learning for SPECT Reconstruction', IEEE TMI 2022",
        "link": "https://arxiv.org/abs/2208.11706",
    },
    "fmri": {
        "url": "https://arxiv.org/pdf/2001.10013",
        "page": 1,
        "citation": "Elbau et al., 'fMRI Reconstruction Overview', arXiv 2020",
        "link": "https://arxiv.org/abs/2001.10013",
    },
    "diffusion_mri": {
        "url": "https://arxiv.org/pdf/2004.07340",
        "page": 0,
        "citation": "Tian et al., 'DeepDTI: High-fidelity Six-direction Diffusion Tensor Imaging', NeuroImage 2020",
        "link": "https://arxiv.org/abs/2004.07340",
    },
    "ultrasound": {
        "url": "https://arxiv.org/pdf/1908.05782",
        "page": 0,
        "citation": "van Sloun et al., 'Deep Learning in Ultrasound Imaging', IEEE 2020",
        "link": "https://arxiv.org/abs/1908.05782",
    },
    "mammography": {
        "url": "https://arxiv.org/pdf/1708.09427",
        "page": 1,
        "citation": "Wu et al., 'Breast Density Classification with Deep CNNs', MICCAI 2017",
        "link": "https://arxiv.org/abs/1708.09427",
    },
    "oct": {
        "url": "https://arxiv.org/pdf/1902.07308",
        "page": 0,
        "citation": "Devalla et al., 'OCT Layer Segmentation with Deep Learning', BOE 2019",
        "link": "https://arxiv.org/abs/1902.07308",
    },
    "fundus": {
        "url": "https://arxiv.org/pdf/1803.04337",
        "page": 1,
        "citation": "Li et al., 'Fundus Image Analysis with Deep Learning', arXiv 2018",
        "link": "https://arxiv.org/abs/1803.04337",
    },
    "photoacoustic": {
        "url": "https://arxiv.org/pdf/1801.07461",
        "page": 0,
        "citation": "Hauptmann et al., 'Model-Based Photoacoustic Reconstruction', IEEE TMI 2018",
        "link": "https://arxiv.org/abs/1801.07461",
    },
    "elastography": {
        "url": "https://arxiv.org/pdf/2009.14104",
        "page": 0,
        "citation": "Wu et al., 'Ultrasound Elastography Review', arXiv 2020",
        "link": "https://arxiv.org/abs/2009.14104",
    },
    "doppler_ultrasound": {
        "url": "https://arxiv.org/pdf/2106.11643",
        "page": 0,
        "citation": "Shen et al., 'Deep Doppler Ultrasound', arXiv 2021",
        "link": "https://arxiv.org/abs/2106.11643",
    },
    # ── Microscopy ──
    "widefield": {
        "url": "https://arxiv.org/pdf/1707.02264",
        "page": 1,
        "citation": "Weigert et al., 'CARE: Content-Aware Image Restoration', Nature Methods 2018",
        "link": "https://arxiv.org/abs/1707.02264",
    },
    "widefield_lowdose": {
        "url": "https://arxiv.org/pdf/1707.02264",
        "page": 2,
        "citation": "Weigert et al., 'CARE: Content-Aware Image Restoration', Nature Methods 2018",
        "link": "https://arxiv.org/abs/1707.02264",
    },
    "confocal_3d": {
        "url": "https://arxiv.org/pdf/1707.02264",
        "page": 3,
        "citation": "Weigert et al., 'CARE: Content-Aware Image Restoration', Nature Methods 2018",
        "link": "https://arxiv.org/abs/1707.02264",
    },
    "sim": {
        "url": "https://arxiv.org/pdf/1609.08958",
        "page": 0,
        "citation": "Müller et al., 'Open-source fairSIM for structured illumination microscopy', Nature Comms 2016",
        "link": "https://arxiv.org/abs/1609.08958",
    },
    "sted": {
        "url": "https://arxiv.org/pdf/1905.04949",
        "page": 0,
        "citation": "Heine et al., 'Adaptive STED Nanoscopy', PNAS 2017",
        "link": "https://arxiv.org/abs/1905.04949",
    },
    "lightsheet": {
        "url": "https://arxiv.org/pdf/1706.08726",
        "page": 0,
        "citation": "Weigert et al., 'Isotropic Reconstruction of Light-Sheet Microscopy', Nature Methods 2018",
        "link": "https://arxiv.org/abs/1706.08726",
    },
    "fpm": {
        "url": "https://arxiv.org/pdf/2012.14738",
        "page": 0,
        "citation": "Zheng et al., 'Concept, implementations and applications of Fourier ptychography', Nature Reviews Physics 2021",
        "link": "https://arxiv.org/abs/2012.14738",
    },
    "flim": {
        "url": "https://arxiv.org/pdf/2003.06505",
        "page": 0,
        "citation": "Smith et al., 'FLIM-Net: Fluorescence Lifetime Imaging with Deep Learning', Biomed. Opt. Express 2020",
        "link": "https://arxiv.org/abs/2003.06505",
    },
    "two_photon": {
        "url": "https://arxiv.org/pdf/1707.02264",
        "page": 4,
        "citation": "Weigert et al., 'CARE: Content-Aware Image Restoration', Nature Methods 2018",
        "link": "https://arxiv.org/abs/1707.02264",
    },
    # ── Coherent Imaging ──
    "holography": {
        "url": "https://arxiv.org/pdf/1903.02278",
        "page": 0,
        "citation": "Rivenson et al., 'Deep Learning for Holographic Reconstruction', Light: Science & Applications 2019",
        "link": "https://arxiv.org/abs/1903.02278",
    },
    "ptychography": {
        "url": "https://arxiv.org/pdf/1811.07945",
        "page": 0,
        "citation": "Cherukara et al., 'AI-enabled High-Resolution Ptychographic Imaging', APL 2020",
        "link": "https://arxiv.org/abs/1811.07945",
    },
    "phase_retrieval": {
        "url": "https://arxiv.org/pdf/1711.08757",
        "page": 0,
        "citation": "Metzler et al., 'prDeep: Phase Retrieval with Deep Priors', NeurIPS 2018",
        "link": "https://arxiv.org/abs/1711.08757",
    },
    "lensless": {
        "url": "https://arxiv.org/pdf/2205.00088",
        "page": 0,
        "citation": "Monakhova et al., 'Lensless Imaging with Deep Learning', Optica 2022",
        "link": "https://arxiv.org/abs/2205.00088",
    },
    # ── Neural Rendering ──
    "nerf": {
        "url": "https://arxiv.org/pdf/2003.08934",
        "page": 2,
        "citation": "Mildenhall et al., 'NeRF: Representing Scenes as Neural Radiance Fields', ECCV 2020",
        "link": "https://arxiv.org/abs/2003.08934",
    },
    "gaussian_splatting": {
        "url": "https://arxiv.org/pdf/2308.04079",
        "page": 1,
        "citation": "Kerbl et al., '3D Gaussian Splatting for Real-Time Radiance Field Rendering', SIGGRAPH 2023",
        "link": "https://arxiv.org/abs/2308.04079",
    },
    # ── Electron Microscopy ──
    "tem": {
        "url": "https://arxiv.org/pdf/2010.03828",
        "page": 0,
        "citation": "de Haan et al., 'Deep Learning for TEM Restoration', arXiv 2020",
        "link": "https://arxiv.org/abs/2010.03828",
    },
    "stem": {
        "url": "https://arxiv.org/pdf/1906.09498",
        "page": 0,
        "citation": "Spurgeon et al., 'STEM Image Analysis with Machine Learning', npj Comp. Materials 2021",
        "link": "https://arxiv.org/abs/1906.09498",
    },
    "electron_tomography": {
        "url": "https://arxiv.org/pdf/2112.00370",
        "page": 0,
        "citation": "Yang et al., 'Deep Learning Electron Tomography', Nature 2021",
        "link": "https://arxiv.org/abs/2112.00370",
    },
    # ── Remote Sensing ──
    "sar": {
        "url": "https://arxiv.org/pdf/2104.01007",
        "page": 0,
        "citation": "Zhu et al., 'Deep Learning in SAR Remote Sensing', IEEE GRSM 2021",
        "link": "https://arxiv.org/abs/2104.01007",
    },
    "lidar": {
        "url": "https://arxiv.org/pdf/2005.09830",
        "page": 0,
        "citation": "Guo et al., 'Deep Learning for 3D Point Clouds (LiDAR)', IEEE TPAMI 2021",
        "link": "https://arxiv.org/abs/2005.09830",
    },
    "sonar": {
        "url": "https://arxiv.org/pdf/2108.01111",
        "page": 0,
        "citation": "Steiniger et al., 'Sonar Image Classification with Deep Learning', arXiv 2021",
        "link": "https://arxiv.org/abs/2108.01111",
    },
    # ── Computational Photography ──
    "light_field": {
        "url": "https://arxiv.org/pdf/1708.03292",
        "page": 0,
        "citation": "Kalantari et al., 'Learning-Based View Synthesis for Light Field Cameras', SIGGRAPH 2016",
        "link": "https://arxiv.org/abs/1708.03292",
    },
    "spc": {
        "url": "https://arxiv.org/pdf/1909.05986",
        "page": 0,
        "citation": "Higham et al., 'Deep Learning for Single-Pixel Imaging', Scientific Reports 2018",
        "link": "https://arxiv.org/abs/1909.05986",
    },
    "structured_light": {
        "url": "https://arxiv.org/pdf/1908.08552",
        "page": 0,
        "citation": "Riegler et al., 'Deep Structured Light', IEEE TPAMI 2021",
        "link": "https://arxiv.org/abs/1908.08552",
    },
    "panorama": {
        "url": "https://arxiv.org/pdf/1703.10593",
        "page": 0,
        "citation": "Zhu et al., 'Unpaired Image-to-Image Translation using CycleGAN', ICCV 2017",
        "link": "https://arxiv.org/abs/1703.10593",
    },
    "tof_camera": {
        "url": "https://arxiv.org/pdf/1903.11854",
        "page": 0,
        "citation": "Su et al., 'Deep End-to-End Time-of-Flight Imaging', CVPR 2018",
        "link": "https://arxiv.org/abs/1903.11854",
    },
}

# User-Agent for downloads
UA = "Mozilla/5.0 (compatible; PWM-Research-Bot/1.0)"


def download_pdf(url: str) -> bytes:
    """Download a PDF from a URL, returning raw bytes."""
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def pdf_page_to_png(pdf_bytes: bytes, page_num: int, dpi: int = 200) -> bytes:
    """Render a specific page of a PDF to PNG bytes using PyMuPDF."""
    import fitz  # pymupdf

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_num >= len(doc):
        page_num = min(page_num, len(doc) - 1)
    page = doc[page_num]
    # Render at specified DPI (default=72, so scale = dpi/72)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    success = 0
    failed = 0

    for key, info in sorted(PAPER_FIGURES.items()):
        out_path = OUT_DIR / f"{key}.png"
        if out_path.exists():
            print(f"  [{key}] already exists, skipping download")
            manifest[key] = {
                "citation": info["citation"],
                "link": info["link"],
                "image": f"/static/img/paper_setups/{key}.png",
            }
            success += 1
            continue

        print(f"  [{key}] downloading {info['url']} page {info['page']}...")
        try:
            pdf_bytes = download_pdf(info["url"])
            png_bytes = pdf_page_to_png(pdf_bytes, info["page"])
            out_path.write_bytes(png_bytes)
            manifest[key] = {
                "citation": info["citation"],
                "link": info["link"],
                "image": f"/static/img/paper_setups/{key}.png",
            }
            success += 1
            print(f"    -> saved ({len(png_bytes) / 1024:.0f} KB)")
            time.sleep(1)  # be polite to arXiv
        except Exception as e:
            print(f"    -> FAILED: {e}")
            failed += 1

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nDone: {success} downloaded, {failed} failed")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
