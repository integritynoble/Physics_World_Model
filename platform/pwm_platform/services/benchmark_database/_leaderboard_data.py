"""Leaderboard data keyed by variant — existing 4 variants have InverseNet data; new variants start empty."""

from __future__ import annotations

# fmt: off

LEADERBOARD_DATA: dict[str, dict[str, list[dict]]] = {

    # ══════════════════════════════════════════════════════════════════════════
    #  SD-CASSI — InverseNet validated results (10 KAIST scenes, 4 algorithms)
    # ══════════════════════════════════════════════════════════════════════════

    "sd_cassi": {
        # B2 leaderboard — Scenario I (Ideal: no mismatch, no noise)
        "b2": [
            {"rank": 1, "method": "MST-L",      "psnr": 34.81, "ssim": 0.973, "sam": 7.44,  "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HDNet",       "psnr": 34.66, "ssim": 0.970, "sam": 6.67,  "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-HSICNN",  "psnr": 25.12, "ssim": 0.758, "sam": 16.10, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "GAP-TV",      "psnr": 24.34, "ssim": 0.723, "sam": 16.66, "source": "InverseNet", "adopted": False},
        ],

        # B2 scenario table — full 3-scenario comparison from InverseNet
        "b2_scenario_table": {
            "description": "10 KAIST scenes (256\u00d7256\u00d728 bands), 4 algorithms, 3 scenarios. Mismatch: mask shift (0.5/0.3 px), rotation (0.1\u00b0), dispersion slope (2.02), axis (0.15\u00b0).",
            "scenarios": [
                {"id": "I",   "name": "Ideal",                "color": "green", "description": "No mismatch, no noise \u2014 theoretical upper bound"},
                {"id": "II",  "name": "Mismatch (uncorrected)","color": "red",   "description": "5-parameter mismatch + Poisson-Gaussian noise, no correction"},
                {"id": "III", "name": "Oracle correction",     "color": "blue",  "description": "True mismatch parameters known \u2014 calibration upper bound"},
            ],
            "methods": [
                {
                    "method": "GAP-TV", "type": "Classical", "mask_aware": True, "params": "0",
                    "s1_psnr": 24.34, "s1_ssim": 0.723, "s1_sam": 16.66,
                    "s2_psnr": 20.96, "s2_ssim": 0.612, "s2_sam": 24.27,
                    "s3_psnr": 21.72, "s3_ssim": 0.688, "s3_sam": 25.97,
                },
                {
                    "method": "PnP-HSICNN", "type": "PnP", "mask_aware": True, "params": "0",
                    "s1_psnr": 25.12, "s1_ssim": 0.758, "s1_sam": 16.10,
                    "s2_psnr": 20.40, "s2_ssim": 0.574, "s2_sam": 23.73,
                    "s3_psnr": 23.08, "s3_ssim": 0.702, "s3_sam": 18.66,
                },
                {
                    "method": "HDNet", "type": "Deep Learning", "mask_aware": False, "params": "2.37M",
                    "s1_psnr": 34.66, "s1_ssim": 0.970, "s1_sam": 6.67,
                    "s2_psnr": 21.88, "s2_ssim": 0.756, "s2_sam": 17.03,
                    "s3_psnr": 21.88, "s3_ssim": 0.756, "s3_sam": 17.03,
                },
                {
                    "method": "MST-L", "type": "Transformer", "mask_aware": True, "params": "2.03M",
                    "s1_psnr": 34.81, "s1_ssim": 0.973, "s1_sam": 7.44,
                    "s2_psnr": 20.83, "s2_ssim": 0.744, "s2_sam": 23.92,
                    "s3_psnr": 27.33, "s3_ssim": 0.881, "s3_sam": 11.74,
                },
            ],
            "method_names": ["GAP-TV", "PnP-HSICNN", "HDNet", "MST-L"],
            "per_scene": [
                {"scene": "Scene 01", "psnr_values": [26.49, 24.08, 24.18,  27.43, 23.47, 25.78,  34.95, 24.37, 24.37,  35.29, 23.96, 29.98]},
                {"scene": "Scene 02", "psnr_values": [24.60, 21.89, 22.82,  25.34, 21.56, 23.80,  35.65, 23.26, 23.26,  36.14, 22.21, 28.42]},
                {"scene": "Scene 03", "psnr_values": [25.96, 18.62, 19.68,  26.71, 17.59, 21.83,  35.54, 18.61, 18.61,  35.66, 16.09, 23.57]},
                {"scene": "Scene 04", "psnr_values": [28.36, 23.37, 24.41,  28.80, 22.89, 25.98,  41.63, 23.98, 23.98,  40.05, 21.91, 29.80]},
                {"scene": "Scene 05", "psnr_values": [23.66, 20.39, 21.27,  24.65, 19.36, 22.75,  32.56, 20.22, 20.22,  32.84, 20.28, 26.75]},
                {"scene": "Scene 06", "psnr_values": [22.34, 20.39, 21.00,  23.12, 20.29, 21.80,  34.33, 22.63, 22.63,  34.56, 22.37, 28.67]},
                {"scene": "Scene 07", "psnr_values": [23.51, 20.56, 21.07,  24.94, 19.86, 22.77,  33.27, 20.79, 20.79,  33.80, 19.76, 26.12]},
                {"scene": "Scene 08", "psnr_values": [22.16, 20.57, 21.10,  22.73, 20.30, 21.92,  32.26, 22.73, 22.73,  32.74, 21.33, 27.44]},
                {"scene": "Scene 09", "psnr_values": [23.03, 19.14, 20.61,  24.03, 18.79, 21.82,  34.18, 21.18, 21.18,  34.37, 19.75, 26.71]},
                {"scene": "Scene 10", "psnr_values": [23.27, 20.58, 21.04,  23.47, 19.90, 22.38,  32.22, 21.06, 21.06,  32.63, 20.69, 25.86]},
            ],
        },

        # B4 leaderboard — Scenario III (Oracle correction: true parameters known)
        "b4": [
            {"rank": 1, "method": "MST-L",      "psnr": 27.33, "ssim": 0.881, "sam": 11.74, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "PnP-HSICNN",  "psnr": 23.08, "ssim": 0.702, "sam": 18.66, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "HDNet",       "psnr": 21.88, "ssim": 0.756, "sam": 17.03, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "GAP-TV",      "psnr": 21.72, "ssim": 0.688, "sam": 25.97, "source": "InverseNet", "adopted": False},
        ],

        # B4 scenario table — mismatch degradation and recovery analysis
        "b4_scenario_table": {
            "description": "Recovery analysis: how much PSNR is lost from mismatch (I\u2192II) and how much is recovered with oracle calibration (II\u2192III).",
            "scenarios": [
                {"id": "II",  "name": "Mismatch (uncorrected)","color": "red",   "description": "5-parameter mismatch + noise, no correction"},
                {"id": "III", "name": "Oracle correction",     "color": "blue",  "description": "True mismatch parameters known"},
            ],
            "methods": [
                {
                    "method": "GAP-TV", "type": "Classical", "mask_aware": True,
                    "s2_psnr": 20.96, "s2_ssim": 0.612, "s3_psnr": 21.72, "s3_ssim": 0.688,
                    "gap_degradation": -3.38, "gap_recovery": 0.76, "recovery_pct": 22,
                },
                {
                    "method": "PnP-HSICNN", "type": "PnP", "mask_aware": True,
                    "s2_psnr": 20.40, "s2_ssim": 0.574, "s3_psnr": 23.08, "s3_ssim": 0.702,
                    "gap_degradation": -4.72, "gap_recovery": 2.68, "recovery_pct": 57,
                },
                {
                    "method": "HDNet", "type": "Deep Learning", "mask_aware": False,
                    "s2_psnr": 21.88, "s2_ssim": 0.756, "s3_psnr": 21.88, "s3_ssim": 0.756,
                    "gap_degradation": -12.77, "gap_recovery": 0.00, "recovery_pct": 0,
                },
                {
                    "method": "MST-L", "type": "Transformer", "mask_aware": True,
                    "s2_psnr": 20.83, "s2_ssim": 0.744, "s3_psnr": 27.33, "s3_ssim": 0.881,
                    "gap_degradation": -13.98, "gap_recovery": 6.50, "recovery_pct": 46,
                },
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  CACTI
    # ══════════════════════════════════════════════════════════════════════════

    "cacti": {
        "b2": [
            {"rank": 1, "method": "EfficientSCI",  "psnr": 35.39, "ssim": 0.973, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "ELP-Unfolding", "psnr": 34.09, "ssim": 0.965, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-FFDNet",    "psnr": 29.28, "ssim": 0.910, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "GAP-TV",        "psnr": 26.75, "ssim": 0.870, "source": "InverseNet", "adopted": False},
        ],
        "b4": [
            {"rank": 1, "method": "EfficientSCI",  "psnr": 27.38, "ssim": 0.927, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "ELP-Unfolding", "psnr": 26.50, "ssim": 0.910, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-FFDNet",    "psnr": 20.15, "ssim": 0.650, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "GAP-TV",        "psnr": 14.81, "ssim": 0.303, "source": "InverseNet", "adopted": False},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  SPC — Block
    # ══════════════════════════════════════════════════════════════════════════

    "spc_block": {
        "b2": [
            {"rank": 1, "method": "ISTA-Net",   "psnr": 31.85, "ssim": 0.916, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HATNet",     "psnr": 30.98, "ssim": 0.905, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-DRUNet", "psnr": 30.53, "ssim": 0.895, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "FISTA-TV",   "psnr": 28.06, "ssim": 0.850, "source": "InverseNet", "adopted": False},
        ],
        "b4": [
            {"rank": 1, "method": "ISTA-Net",   "psnr": 27.45, "ssim": 0.760, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HATNet",     "psnr": 26.80, "ssim": 0.745, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-DRUNet", "psnr": 24.10, "ssim": 0.690, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "FISTA-TV",   "psnr": 19.02, "ssim": 0.584, "source": "InverseNet", "adopted": False},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  SPC — Kronecker
    # ══════════════════════════════════════════════════════════════════════════

    "spc_kronecker": {
        "b2": [
            {"rank": 1, "method": "ISTA-Net",   "psnr": 31.85, "ssim": 0.916, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HATNet",     "psnr": 30.98, "ssim": 0.905, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-DRUNet", "psnr": 30.53, "ssim": 0.895, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "FISTA-TV",   "psnr": 28.06, "ssim": 0.850, "source": "InverseNet", "adopted": False},
        ],
        "b4": [
            {"rank": 1, "method": "ISTA-Net",   "psnr": 27.45, "ssim": 0.760, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HATNet",     "psnr": 26.80, "ssim": 0.745, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-DRUNet", "psnr": 24.10, "ssim": 0.690, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "FISTA-TV",   "psnr": 19.02, "ssim": 0.584, "source": "InverseNet", "adopted": False},
        ],
    },
}
