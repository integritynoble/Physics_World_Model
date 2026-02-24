"""
Benchmark Database — static data for modality variant benchmark pages.

Contains spec primitives, variant-specific benchmark configurations (B1-B4),
leaderboard data sourced from the InverseNet paper, flowcharts, and credits
configuration. Pattern follows modality_database.py (pure data, no DB models).
"""

from __future__ import annotations

# ── Spec Primitives (the 11 building blocks of any imaging forward model) ────

SPEC_PRIMITIVES = {
    "P": {
        "symbol": "P",
        "name": "Propagation",
        "description": "Free-space or medium propagation kernel (Fresnel, Rayleigh-Sommerfeld).",
    },
    "M": {
        "symbol": "M",
        "name": "Mask / Modulation",
        "description": "Spatial or spatio-temporal amplitude modulation (coded aperture, SLM pattern).",
    },
    "Pi": {
        "symbol": "\u03a0",
        "name": "Projection",
        "description": "Geometric projection operator (Radon transform, fan-beam, cone-beam).",
    },
    "F": {
        "symbol": "F",
        "name": "Fourier Sampling",
        "description": "Sampling in the Fourier / k-space domain (MRI, ptychography).",
    },
    "C": {
        "symbol": "C",
        "name": "Convolution",
        "description": "Shift-invariant convolution with a point-spread function (PSF).",
    },
    "Sigma": {
        "symbol": "\u03a3",
        "name": "Summation / Integration",
        "description": "Summation along a physical dimension (spectral, temporal, angular).",
    },
    "D": {
        "symbol": "D",
        "name": "Detector",
        "description": "Sensor readout with gain g and noise model \u03b7 (Gaussian, Poisson, mixed).",
    },
    "S": {
        "symbol": "S",
        "name": "Structured Illumination",
        "description": "Patterned illumination (block, Hadamard, random) applied to the scene.",
    },
    "W": {
        "symbol": "W",
        "name": "Wavelength Dispersion",
        "description": "Spectral dispersion element (prism, grating) with shift \u03b1 and aperture a.",
    },
    "R": {
        "symbol": "R",
        "name": "Rotation / Motion",
        "description": "Sample or gantry rotation (CT, electron tomography).",
    },
    "Lambda": {
        "symbol": "\u039b",
        "name": "Wavelength Selection",
        "description": "Spectral filter or monochromator selecting a wavelength band.",
    },
}

# ── Variant Database ─────────────────────────────────────────────────────────

VARIANT_DATABASE = {
    # ── SD-CASSI ──────────────────────────────────────────────────────────
    "sd_cassi": {
        "display_name": "SD-CASSI",
        "full_name": "Single-Disperser Coded Aperture Snapshot Spectral Imager",
        "parent_modality": "cassi",
        "category": "compressive",
        "spec_notation": "M(mask) \u2192 W(\u03b1, a) \u2192 \u03a3_\u03bb \u2192 D(g, \u03b7\u2084)",
        "spec_dag": [
            {"primitive": "M", "params": "mask", "label": "Coded Aperture"},
            {"primitive": "W", "params": "\u03b1, a", "label": "Prism Dispersion"},
            {"primitive": "Sigma", "params": "\u03bb", "label": "Spectral Sum"},
            {"primitive": "D", "params": "g, \u03b7\u2084", "label": "Detector"},
        ],
        "mismatch_params": [
            {"name": "mask_dx", "symbol": "\u0394x", "description": "Mask lateral shift (pixels)", "nominal": 0, "perturbed": 0.5},
            {"name": "mask_dy", "symbol": "\u0394y", "description": "Mask vertical shift (pixels)", "nominal": 0, "perturbed": 0.3},
            {"name": "mask_theta", "symbol": "\u03b8", "description": "Mask rotation (rad)", "nominal": 0, "perturbed": 0.1},
            {"name": "disp_a1", "symbol": "a\u2081", "description": "Dispersion coefficient", "nominal": 2.0, "perturbed": 2.02},
            {"name": "disp_alpha", "symbol": "\u03b1", "description": "Dispersion angle (rad)", "nominal": 0, "perturbed": 0.15},
        ],
        "benchmarks": [
            {
                "id": "B1",
                "title": "B1 \u2014 LLM Spec Router",
                "description": "Given a natural-language modality description, the LLM selects the correct spec from the primitive library. Scored on spec-match accuracy.",
                "input": "Natural-language modality description",
                "output": "Selected spec (primitive DAG)",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SD-CASSI B1 Spec Prompts (Public)",
                    "description": "50 natural-language imaging system descriptions paired with their correct spec DAGs from the primitive library.",
                    "format": "JSON",
                    "num_samples": 50,
                    "size_mb": 0.1,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/sd_cassi_b1_public.json",
                },
                "hidden_dataset": {
                    "name": "SD-CASSI B1 Spec Prompts (Hidden)",
                    "description": "100 held-out natural-language descriptions used for server-side evaluation of spec-match accuracy.",
                    "format": "JSON",
                    "num_samples": 100,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [],
                "links": {"contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-spec.md"},
                "credits": None,
            },
            {
                "id": "B2",
                "title": "B2 \u2014 Algorithm Correction + Reconstruction",
                "description": "Given a mismatched forward model and measurements, the algorithm corrects the spec and reconstructs the signal. Scored on PSNR / SSIM.",
                "input": "Measurements y, mismatched forward model H\u0303",
                "output": "Reconstructed signal x\u0302",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SD-CASSI B2 Recon Data (Public)",
                    "description": "5 KAIST spectral scenes (256\u00d7256\u00d728 bands) with measurements y, mismatched forward model H\u0303, and ground-truth signal x.",
                    "format": "HDF5",
                    "num_samples": 5,
                    "size_mb": 45,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/sd_cassi_b2_public.h5",
                },
                "hidden_dataset": {
                    "name": "SD-CASSI B2 Recon Data (Hidden)",
                    "description": "Remaining KAIST scenes held out for server-side PSNR/SSIM evaluation.",
                    "format": "HDF5",
                    "num_samples": 5,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [
                    {"rank": 1, "method": "MST-L", "psnr": 34.81, "ssim": 0.973, "source": "InverseNet", "adopted": True},
                    {"rank": 2, "method": "HDNet", "psnr": 34.66, "ssim": 0.970, "source": "InverseNet", "adopted": False},
                    {"rank": 3, "method": "PnP-HSICNN", "psnr": 25.12, "ssim": 0.832, "source": "InverseNet", "adopted": False},
                    {"rank": 4, "method": "GAP-TV", "psnr": 24.34, "ssim": 0.815, "source": "InverseNet", "adopted": False},
                ],
                "links": {
                    "contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-dataset.md",
                    "submit_algorithm": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-algorithm.md",
                },
                "credits": {"winner_share_pct": 30, "pool_source": "platform_profit"},
            },
            {
                "id": "B3",
                "title": "B3 \u2014 LLM Spec Router (with ground-truth)",
                "description": "Given real measurements y, true forward model H, and a candidate spec, the LLM evaluates whether the spec matches the true system. Scored on classification accuracy.",
                "input": "Measurements y, true forward model H, candidate spec",
                "output": "Match / no-match classification",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SD-CASSI B3 Spec Validation (Public)",
                    "description": "50 triplets of {measurements y, true forward model H, candidate spec} with match/no-match labels for spec validation.",
                    "format": "JSON + HDF5",
                    "num_samples": 50,
                    "size_mb": 12,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/sd_cassi_b3_public.tar.gz",
                },
                "hidden_dataset": {
                    "name": "SD-CASSI B3 Spec Validation (Hidden)",
                    "description": "100 held-out triplets used for server-side evaluation of spec classification accuracy.",
                    "format": "JSON + HDF5",
                    "num_samples": 100,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [],
                "links": {"contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-spec.md"},
                "credits": None,
            },
            {
                "id": "B4",
                "title": "B4 \u2014 Algorithm Reconstruction (with drift)",
                "description": "Given measurements from a drifted system (low-scoring B3 specs), the algorithm reconstructs the signal. Tests robustness to forward-model drift.",
                "input": "Measurements y, drifted forward model H\u0302",
                "output": "Reconstructed signal x\u0302",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SD-CASSI B4 Drift Recon Data (Public)",
                    "description": "5 KAIST spectral scenes (256\u00d7256\u00d728 bands) with measurements from Scenario III drifted system, drifted forward model H\u0302, and ground-truth signal x.",
                    "format": "HDF5",
                    "num_samples": 5,
                    "size_mb": 45,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/sd_cassi_b4_public.h5",
                },
                "hidden_dataset": {
                    "name": "SD-CASSI B4 Drift Recon Data (Hidden)",
                    "description": "Remaining drifted KAIST scenes held out for server-side PSNR/SSIM evaluation.",
                    "format": "HDF5",
                    "num_samples": 5,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [
                    {"rank": 1, "method": "MST-L", "psnr": 27.33, "ssim": 0.881, "source": "InverseNet", "adopted": True},
                    {"rank": 2, "method": "HDNet", "psnr": 27.10, "ssim": 0.875, "source": "InverseNet", "adopted": False},
                    {"rank": 3, "method": "PnP-HSICNN", "psnr": 22.45, "ssim": 0.780, "source": "InverseNet", "adopted": False},
                    {"rank": 4, "method": "GAP-TV", "psnr": 20.83, "ssim": 0.744, "source": "InverseNet", "adopted": False},
                ],
                "links": {
                    "contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-dataset.md",
                    "submit_algorithm": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-algorithm.md",
                },
                "credits": {"winner_share_pct": 30, "pool_source": "platform_profit"},
            },
        ],
        "credits_config": {
            "profit_pool_pct": 40,
            "winner_share_pct": 30,
            "min_withdrawal_usd": 100,
        },
    },

    # ── CACTI ─────────────────────────────────────────────────────────────
    "cacti": {
        "display_name": "CACTI",
        "full_name": "Coded Aperture Compressive Temporal Imaging",
        "parent_modality": "cacti",
        "category": "compressive",
        "spec_notation": "M(m_t) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2084)",
        "spec_dag": [
            {"primitive": "M", "params": "m_t", "label": "Temporal Mask"},
            {"primitive": "Sigma", "params": "t", "label": "Temporal Sum"},
            {"primitive": "D", "params": "g, \u03b7\u2084", "label": "Detector"},
        ],
        "mismatch_params": [
            {"name": "mask_dx", "symbol": "\u0394x", "description": "Mask lateral shift (pixels)", "nominal": 0, "perturbed": 0.5},
            {"name": "mask_dy", "symbol": "\u0394y", "description": "Mask vertical shift (pixels)", "nominal": 0, "perturbed": 0.3},
            {"name": "mask_theta", "symbol": "\u03b8", "description": "Mask rotation (rad)", "nominal": 0, "perturbed": 0.1},
            {"name": "clock_offset", "symbol": "\u0394t", "description": "Clock synchronization offset", "nominal": 0, "perturbed": 0.05},
            {"name": "duty_cycle", "symbol": "d", "description": "Shutter duty cycle", "nominal": 1.0, "perturbed": 0.95},
            {"name": "gain", "symbol": "g", "description": "Detector gain multiplier", "nominal": 1.0, "perturbed": 1.02},
        ],
        "benchmarks": [
            {
                "id": "B1",
                "title": "B1 \u2014 LLM Spec Router",
                "description": "Given a natural-language modality description, the LLM selects the correct spec from the primitive library. Scored on spec-match accuracy.",
                "input": "Natural-language modality description",
                "output": "Selected spec (primitive DAG)",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "CACTI B1 Spec Prompts (Public)",
                    "description": "50 natural-language imaging system descriptions paired with their correct spec DAGs from the primitive library.",
                    "format": "JSON",
                    "num_samples": 50,
                    "size_mb": 0.1,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/cacti_b1_public.json",
                },
                "hidden_dataset": {
                    "name": "CACTI B1 Spec Prompts (Hidden)",
                    "description": "100 held-out natural-language descriptions used for server-side evaluation of spec-match accuracy.",
                    "format": "JSON",
                    "num_samples": 100,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [],
                "links": {"contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-spec.md"},
                "credits": None,
            },
            {
                "id": "B2",
                "title": "B2 \u2014 Algorithm Correction + Reconstruction",
                "description": "Given a mismatched forward model and measurements, the algorithm corrects the spec and reconstructs the signal. Scored on PSNR / SSIM.",
                "input": "Measurements y, mismatched forward model H\u0303",
                "output": "Reconstructed signal x\u0302",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "CACTI B2 Recon Data (Public)",
                    "description": "3 video sequences (256\u00d7256\u00d78 frames) with measurements y, mismatched forward model H\u0303, and ground-truth signal x.",
                    "format": "HDF5",
                    "num_samples": 3,
                    "size_mb": 30,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/cacti_b2_public.h5",
                },
                "hidden_dataset": {
                    "name": "CACTI B2 Recon Data (Hidden)",
                    "description": "Remaining video sequences held out for server-side PSNR/SSIM evaluation.",
                    "format": "HDF5",
                    "num_samples": 3,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [
                    {"rank": 1, "method": "EfficientSCI", "psnr": 35.39, "ssim": 0.973, "source": "InverseNet", "adopted": True},
                    {"rank": 2, "method": "ELP-Unfolding", "psnr": 34.09, "ssim": 0.965, "source": "InverseNet", "adopted": False},
                    {"rank": 3, "method": "PnP-FFDNet", "psnr": 29.28, "ssim": 0.910, "source": "InverseNet", "adopted": False},
                    {"rank": 4, "method": "GAP-TV", "psnr": 26.75, "ssim": 0.870, "source": "InverseNet", "adopted": False},
                ],
                "links": {
                    "contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-dataset.md",
                    "submit_algorithm": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-algorithm.md",
                },
                "credits": {"winner_share_pct": 30, "pool_source": "platform_profit"},
            },
            {
                "id": "B3",
                "title": "B3 \u2014 LLM Spec Router (with ground-truth)",
                "description": "Given real measurements y, true forward model H, and a candidate spec, the LLM evaluates whether the spec matches the true system. Scored on classification accuracy.",
                "input": "Measurements y, true forward model H, candidate spec",
                "output": "Match / no-match classification",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "CACTI B3 Spec Validation (Public)",
                    "description": "50 triplets of {measurements y, true forward model H, candidate spec} with match/no-match labels for spec validation.",
                    "format": "JSON + HDF5",
                    "num_samples": 50,
                    "size_mb": 8,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/cacti_b3_public.tar.gz",
                },
                "hidden_dataset": {
                    "name": "CACTI B3 Spec Validation (Hidden)",
                    "description": "100 held-out triplets used for server-side evaluation of spec classification accuracy.",
                    "format": "JSON + HDF5",
                    "num_samples": 100,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [],
                "links": {"contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-spec.md"},
                "credits": None,
            },
            {
                "id": "B4",
                "title": "B4 \u2014 Algorithm Reconstruction (with drift)",
                "description": "Given measurements from a drifted system (low-scoring B3 specs), the algorithm reconstructs the signal. Tests robustness to forward-model drift.",
                "input": "Measurements y, drifted forward model H\u0302",
                "output": "Reconstructed signal x\u0302",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "CACTI B4 Drift Recon Data (Public)",
                    "description": "3 video sequences (256\u00d7256\u00d78 frames) with measurements from drifted system, drifted forward model H\u0302, and ground-truth signal x.",
                    "format": "HDF5",
                    "num_samples": 3,
                    "size_mb": 30,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/cacti_b4_public.h5",
                },
                "hidden_dataset": {
                    "name": "CACTI B4 Drift Recon Data (Hidden)",
                    "description": "Remaining drifted video sequences held out for server-side PSNR/SSIM evaluation.",
                    "format": "HDF5",
                    "num_samples": 3,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [
                    {"rank": 1, "method": "EfficientSCI", "psnr": 27.38, "ssim": 0.927, "source": "InverseNet", "adopted": True},
                    {"rank": 2, "method": "ELP-Unfolding", "psnr": 26.50, "ssim": 0.910, "source": "InverseNet", "adopted": False},
                    {"rank": 3, "method": "PnP-FFDNet", "psnr": 20.15, "ssim": 0.650, "source": "InverseNet", "adopted": False},
                    {"rank": 4, "method": "GAP-TV", "psnr": 14.81, "ssim": 0.303, "source": "InverseNet", "adopted": False},
                ],
                "links": {
                    "contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-dataset.md",
                    "submit_algorithm": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-algorithm.md",
                },
                "credits": {"winner_share_pct": 30, "pool_source": "platform_profit"},
            },
        ],
        "credits_config": {
            "profit_pool_pct": 40,
            "winner_share_pct": 30,
            "min_withdrawal_usd": 100,
        },
    },

    # ── SPC-Block ─────────────────────────────────────────────────────────
    "spc_block": {
        "display_name": "SPC-Block",
        "full_name": "Single-Pixel Camera (Block Sensing)",
        "parent_modality": "spc",
        "category": "compressive",
        "spec_notation": "S(block) \u2192 M(\u03a6) \u2192 \u03a3 \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "S", "params": "block", "label": "Block Illumination"},
            {"primitive": "M", "params": "\u03a6", "label": "Sensing Matrix"},
            {"primitive": "Sigma", "params": "", "label": "Spatial Sum"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Detector"},
        ],
        "mismatch_params": [
            {"name": "gain_alpha", "symbol": "\u03b1", "description": "Gain drift coefficient", "nominal": 0, "perturbed": 0.0015},
            {"name": "sigma_y", "symbol": "\u03c3_y", "description": "Measurement noise std", "nominal": 0, "perturbed": 0.03},
        ],
        "benchmarks": [
            {
                "id": "B1",
                "title": "B1 \u2014 LLM Spec Router",
                "description": "Given a natural-language modality description, the LLM selects the correct spec from the primitive library. Scored on spec-match accuracy.",
                "input": "Natural-language modality description",
                "output": "Selected spec (primitive DAG)",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SPC-Block B1 Spec Prompts (Public)",
                    "description": "50 natural-language imaging system descriptions paired with their correct spec DAGs from the primitive library.",
                    "format": "JSON",
                    "num_samples": 50,
                    "size_mb": 0.1,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/spc_block_b1_public.json",
                },
                "hidden_dataset": {
                    "name": "SPC-Block B1 Spec Prompts (Hidden)",
                    "description": "100 held-out natural-language descriptions used for server-side evaluation of spec-match accuracy.",
                    "format": "JSON",
                    "num_samples": 100,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [],
                "links": {"contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-spec.md"},
                "credits": None,
            },
            {
                "id": "B2",
                "title": "B2 \u2014 Algorithm Correction + Reconstruction",
                "description": "Given a mismatched forward model and measurements, the algorithm corrects the spec and reconstructs the signal. Scored on PSNR / SSIM.",
                "input": "Measurements y, mismatched forward model H\u0303",
                "output": "Reconstructed signal x\u0302",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SPC-Block B2 Recon Data (Public)",
                    "description": "6 Set11 images (64\u00d764) with measurements y, mismatched forward model H\u0303, and ground-truth signal x.",
                    "format": "HDF5",
                    "num_samples": 6,
                    "size_mb": 5,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/spc_block_b2_public.h5",
                },
                "hidden_dataset": {
                    "name": "SPC-Block B2 Recon Data (Hidden)",
                    "description": "Remaining Set11 images held out for server-side PSNR/SSIM evaluation.",
                    "format": "HDF5",
                    "num_samples": 5,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [
                    {"rank": 1, "method": "ISTA-Net", "psnr": 31.85, "ssim": 0.916, "source": "InverseNet", "adopted": True},
                    {"rank": 2, "method": "HATNet", "psnr": 30.98, "ssim": 0.905, "source": "InverseNet", "adopted": False},
                    {"rank": 3, "method": "PnP-DRUNet", "psnr": 30.53, "ssim": 0.895, "source": "InverseNet", "adopted": False},
                    {"rank": 4, "method": "FISTA-TV", "psnr": 28.06, "ssim": 0.850, "source": "InverseNet", "adopted": False},
                ],
                "links": {
                    "contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-dataset.md",
                    "submit_algorithm": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-algorithm.md",
                },
                "credits": {"winner_share_pct": 30, "pool_source": "platform_profit"},
            },
            {
                "id": "B3",
                "title": "B3 \u2014 LLM Spec Router (with ground-truth)",
                "description": "Given real measurements y, true forward model H, and a candidate spec, the LLM evaluates whether the spec matches the true system. Scored on classification accuracy.",
                "input": "Measurements y, true forward model H, candidate spec",
                "output": "Match / no-match classification",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SPC-Block B3 Spec Validation (Public)",
                    "description": "50 triplets of {measurements y, true forward model H, candidate spec} with match/no-match labels for spec validation.",
                    "format": "JSON + HDF5",
                    "num_samples": 50,
                    "size_mb": 3,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/spc_block_b3_public.tar.gz",
                },
                "hidden_dataset": {
                    "name": "SPC-Block B3 Spec Validation (Hidden)",
                    "description": "100 held-out triplets used for server-side evaluation of spec classification accuracy.",
                    "format": "JSON + HDF5",
                    "num_samples": 100,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [],
                "links": {"contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-spec.md"},
                "credits": None,
            },
            {
                "id": "B4",
                "title": "B4 \u2014 Algorithm Reconstruction (with drift)",
                "description": "Given measurements from a drifted system (low-scoring B3 specs), the algorithm reconstructs the signal. Tests robustness to forward-model drift.",
                "input": "Measurements y, drifted forward model H\u0302",
                "output": "Reconstructed signal x\u0302",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SPC-Block B4 Drift Recon Data (Public)",
                    "description": "6 Set11 images (64\u00d764) with measurements from drifted system, drifted forward model H\u0302, and ground-truth signal x.",
                    "format": "HDF5",
                    "num_samples": 6,
                    "size_mb": 5,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/spc_block_b4_public.h5",
                },
                "hidden_dataset": {
                    "name": "SPC-Block B4 Drift Recon Data (Hidden)",
                    "description": "Remaining drifted Set11 images held out for server-side PSNR/SSIM evaluation.",
                    "format": "HDF5",
                    "num_samples": 5,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [
                    {"rank": 1, "method": "ISTA-Net", "psnr": 27.45, "ssim": 0.760, "source": "InverseNet", "adopted": True},
                    {"rank": 2, "method": "HATNet", "psnr": 26.80, "ssim": 0.745, "source": "InverseNet", "adopted": False},
                    {"rank": 3, "method": "PnP-DRUNet", "psnr": 24.10, "ssim": 0.690, "source": "InverseNet", "adopted": False},
                    {"rank": 4, "method": "FISTA-TV", "psnr": 19.02, "ssim": 0.584, "source": "InverseNet", "adopted": False},
                ],
                "links": {
                    "contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-dataset.md",
                    "submit_algorithm": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-algorithm.md",
                },
                "credits": {"winner_share_pct": 30, "pool_source": "platform_profit"},
            },
        ],
        "credits_config": {
            "profit_pool_pct": 40,
            "winner_share_pct": 30,
            "min_withdrawal_usd": 100,
        },
    },

    # ── SPC-Kronecker ─────────────────────────────────────────────────────
    "spc_kronecker": {
        "display_name": "SPC-Kronecker",
        "full_name": "Single-Pixel Camera (Kronecker Sensing)",
        "parent_modality": "spc",
        "category": "compressive",
        "spec_notation": "M(H\u2297W) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "M", "params": "H\u2297W", "label": "Kronecker Sensing"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Detector"},
        ],
        "mismatch_params": [
            {"name": "gain_alpha", "symbol": "\u03b1", "description": "Gain drift coefficient", "nominal": 0, "perturbed": 0.0015},
            {"name": "sigma_y", "symbol": "\u03c3_y", "description": "Measurement noise std", "nominal": 0, "perturbed": 0.03},
        ],
        "benchmarks": [
            {
                "id": "B1",
                "title": "B1 \u2014 LLM Spec Router",
                "description": "Given a natural-language modality description, the LLM selects the correct spec from the primitive library. Scored on spec-match accuracy.",
                "input": "Natural-language modality description",
                "output": "Selected spec (primitive DAG)",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SPC-Kronecker B1 Spec Prompts (Public)",
                    "description": "50 natural-language imaging system descriptions paired with their correct spec DAGs from the primitive library.",
                    "format": "JSON",
                    "num_samples": 50,
                    "size_mb": 0.1,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/spc_kronecker_b1_public.json",
                },
                "hidden_dataset": {
                    "name": "SPC-Kronecker B1 Spec Prompts (Hidden)",
                    "description": "100 held-out natural-language descriptions used for server-side evaluation of spec-match accuracy.",
                    "format": "JSON",
                    "num_samples": 100,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [],
                "links": {"contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-spec.md"},
                "credits": None,
            },
            {
                "id": "B2",
                "title": "B2 \u2014 Algorithm Correction + Reconstruction",
                "description": "Given a mismatched forward model and measurements, the algorithm corrects the spec and reconstructs the signal. Scored on PSNR / SSIM.",
                "input": "Measurements y, mismatched forward model H\u0303",
                "output": "Reconstructed signal x\u0302",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SPC-Kronecker B2 Recon Data (Public)",
                    "description": "6 Set11 images (64\u00d764) with measurements y, mismatched forward model H\u0303, and ground-truth signal x.",
                    "format": "HDF5",
                    "num_samples": 6,
                    "size_mb": 5,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/spc_kronecker_b2_public.h5",
                },
                "hidden_dataset": {
                    "name": "SPC-Kronecker B2 Recon Data (Hidden)",
                    "description": "Remaining Set11 images held out for server-side PSNR/SSIM evaluation.",
                    "format": "HDF5",
                    "num_samples": 5,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [
                    {"rank": 1, "method": "ISTA-Net", "psnr": 31.85, "ssim": 0.916, "source": "InverseNet", "adopted": True},
                    {"rank": 2, "method": "HATNet", "psnr": 30.98, "ssim": 0.905, "source": "InverseNet", "adopted": False},
                    {"rank": 3, "method": "PnP-DRUNet", "psnr": 30.53, "ssim": 0.895, "source": "InverseNet", "adopted": False},
                    {"rank": 4, "method": "FISTA-TV", "psnr": 28.06, "ssim": 0.850, "source": "InverseNet", "adopted": False},
                ],
                "links": {
                    "contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-dataset.md",
                    "submit_algorithm": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-algorithm.md",
                },
                "credits": {"winner_share_pct": 30, "pool_source": "platform_profit"},
            },
            {
                "id": "B3",
                "title": "B3 \u2014 LLM Spec Router (with ground-truth)",
                "description": "Given real measurements y, true forward model H, and a candidate spec, the LLM evaluates whether the spec matches the true system. Scored on classification accuracy.",
                "input": "Measurements y, true forward model H, candidate spec",
                "output": "Match / no-match classification",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SPC-Kronecker B3 Spec Validation (Public)",
                    "description": "50 triplets of {measurements y, true forward model H, candidate spec} with match/no-match labels for spec validation.",
                    "format": "JSON + HDF5",
                    "num_samples": 50,
                    "size_mb": 3,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/spc_kronecker_b3_public.tar.gz",
                },
                "hidden_dataset": {
                    "name": "SPC-Kronecker B3 Spec Validation (Hidden)",
                    "description": "100 held-out triplets used for server-side evaluation of spec classification accuracy.",
                    "format": "JSON + HDF5",
                    "num_samples": 100,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [],
                "links": {"contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-spec.md"},
                "credits": None,
            },
            {
                "id": "B4",
                "title": "B4 \u2014 Algorithm Reconstruction (with drift)",
                "description": "Given measurements from a drifted system (low-scoring B3 specs), the algorithm reconstructs the signal. Tests robustness to forward-model drift.",
                "input": "Measurements y, drifted forward model H\u0302",
                "output": "Reconstructed signal x\u0302",
                "has_public_dataset": True,
                "has_hidden_dataset": True,
                "public_dataset": {
                    "name": "SPC-Kronecker B4 Drift Recon Data (Public)",
                    "description": "6 Set11 images (64\u00d764) with measurements from drifted system, drifted forward model H\u0302, and ground-truth signal x.",
                    "format": "HDF5",
                    "num_samples": 6,
                    "size_mb": 5,
                    "download_url": "https://github.com/InverseNet/benchmark-data/releases/download/v1.0/spc_kronecker_b4_public.h5",
                },
                "hidden_dataset": {
                    "name": "SPC-Kronecker B4 Drift Recon Data (Hidden)",
                    "description": "Remaining drifted Set11 images held out for server-side PSNR/SSIM evaluation.",
                    "format": "HDF5",
                    "num_samples": 5,
                    "note": "Hidden test set \u2014 used for leaderboard scoring only.",
                },
                "leaderboard": [
                    {"rank": 1, "method": "ISTA-Net", "psnr": 27.45, "ssim": 0.760, "source": "InverseNet", "adopted": True},
                    {"rank": 2, "method": "HATNet", "psnr": 26.80, "ssim": 0.745, "source": "InverseNet", "adopted": False},
                    {"rank": 3, "method": "PnP-DRUNet", "psnr": 24.10, "ssim": 0.690, "source": "InverseNet", "adopted": False},
                    {"rank": 4, "method": "FISTA-TV", "psnr": 19.02, "ssim": 0.584, "source": "InverseNet", "adopted": False},
                ],
                "links": {
                    "contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-dataset.md",
                    "submit_algorithm": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-algorithm.md",
                },
                "credits": {"winner_share_pct": 30, "pool_source": "platform_profit"},
            },
        ],
        "credits_config": {
            "profit_pool_pct": 40,
            "winner_share_pct": 30,
            "min_withdrawal_usd": 100,
        },
    },
}


# ── Flowcharts ────────────────────────────────────────────────────────────────

FLOWCHARTS = {
    "flow_b1_b2": {
        "title": "Flow 1: Spec Selection + Reconstruction (B1 \u2192 B2)",
        "description": "User describes imaging system in natural language; LLM routes to spec; algorithm reconstructs.",
        "steps": [
            {"label": "User Prompt", "detail": "Natural-language description of imaging modality", "color": "blue"},
            {"label": "B1: LLM Spec Router", "detail": "Selects spec from 11 primitives", "color": "indigo"},
            {"label": "Spec Correct?", "detail": "Validate against known specs", "color": "amber"},
            {"label": "Refine Prompt", "detail": "User clarifies if spec mismatch", "color": "amber"},
            {"label": "B2: Algorithm Recon", "detail": "Corrects forward model + reconstructs x\u0302", "color": "indigo"},
            {"label": "Output x\u0302", "detail": "Reconstructed signal with PSNR/SSIM", "color": "green"},
        ],
    },
    "flow_b3_b4": {
        "title": "Flow 2: Ground-Truth Validation + Drift Robustness (B3 \u2192 B4)",
        "description": "User uploads measurements and true forward model; LLM validates spec; algorithm handles drift.",
        "steps": [
            {"label": "Upload y, H", "detail": "User provides measurements + true forward model", "color": "blue"},
            {"label": "B3: LLM Spec Validator", "detail": "Compares candidate spec vs true spec", "color": "indigo"},
            {"label": "Spec Match?", "detail": "Score spec against ground-truth H", "color": "amber"},
            {"label": "Flag Drift", "detail": "Low-scoring specs indicate system drift", "color": "amber"},
            {"label": "B4: Algorithm Recon", "detail": "Reconstructs from drifted forward model", "color": "indigo"},
            {"label": "Output x\u0302", "detail": "Drift-robust reconstruction", "color": "green"},
        ],
    },
}


# ── Mapping: parent modality key → variant keys ──────────────────────────────

_MODALITY_TO_VARIANTS = {
    "cassi": ["sd_cassi"],
    "cacti": ["cacti"],
    "spc": ["spc_block", "spc_kronecker"],
}


# ── Public API ────────────────────────────────────────────────────────────────


def get_variant(variant_key: str) -> dict | None:
    """Return full variant record, or None if not found."""
    entry = VARIANT_DATABASE.get(variant_key)
    if entry is None:
        return None
    return dict(entry)


def list_all_variant_keys() -> list[str]:
    """Return all variant keys in insertion order."""
    return list(VARIANT_DATABASE.keys())


def list_variants_for_modality(modality_key: str) -> list[str]:
    """Return variant keys belonging to a parent modality (e.g. 'spc' → ['spc_block', 'spc_kronecker'])."""
    return list(_MODALITY_TO_VARIANTS.get(modality_key, []))


def get_flowcharts() -> dict:
    """Return the flowchart definitions."""
    return dict(FLOWCHARTS)


def get_spec_primitives() -> dict:
    """Return the 11 spec primitives."""
    return dict(SPEC_PRIMITIVES)
