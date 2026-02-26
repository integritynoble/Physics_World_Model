"""Leaderboard data keyed by variant — InverseNet validated baselines for CASSI, CACTI, SPC."""

from __future__ import annotations

# fmt: off

LEADERBOARD_DATA: dict[str, dict[str, list[dict]]] = {

    # ══════════════════════════════════════════════════════════════════════════
    #  SD-CASSI — InverseNet validated results (10 KAIST scenes, 4 algorithms)
    # ══════════════════════════════════════════════════════════════════════════

    "sd_cassi": {
        # Challenge leaderboard — Blind Reconstruction Challenge (3-tier)
        # Scores: 0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × consistency; overall = mean(public, dev, hidden)
        "challenge": [
            {"rank": 1, "method": "MST-L + gradient",     "public_score": 0.751, "dev_score": 0.706, "hidden_score": 0.689, "overall_score": 0.715, "details": {"public": {"psnr": 26.88, "ssim": 0.871, "consistency": 0.94}, "dev": {"psnr": 24.73, "ssim": 0.826, "consistency": 0.89}, "hidden": {"psnr": 24.08, "ssim": 0.812, "consistency": 0.86}}, "source": "InverseNet baseline"},
            {"rank": 2, "method": "HDNet + gradient",      "public_score": 0.653, "dev_score": 0.637, "hidden_score": 0.631, "overall_score": 0.640, "details": {"public": {"psnr": 21.88, "ssim": 0.756, "consistency": 0.88}, "dev": {"psnr": 21.88, "ssim": 0.756, "consistency": 0.80}, "hidden": {"psnr": 21.88, "ssim": 0.756, "consistency": 0.77}}, "source": "InverseNet baseline"},
            {"rank": 3, "method": "PnP-HSICNN + gradient", "public_score": 0.645, "dev_score": 0.608, "hidden_score": 0.594, "overall_score": 0.616, "details": {"public": {"psnr": 22.84, "ssim": 0.690, "consistency": 0.93}, "dev": {"psnr": 21.90, "ssim": 0.646, "consistency": 0.87}, "hidden": {"psnr": 21.63, "ssim": 0.633, "consistency": 0.84}}, "source": "InverseNet baseline"},
            {"rank": 4, "method": "GAP-TV + gradient",     "public_score": 0.627, "dev_score": 0.600, "hidden_score": 0.590, "overall_score": 0.606, "details": {"public": {"psnr": 21.63, "ssim": 0.679, "consistency": 0.91}, "dev": {"psnr": 21.36, "ssim": 0.652, "consistency": 0.84}, "hidden": {"psnr": 21.28, "ssim": 0.644, "consistency": 0.81}}, "source": "InverseNet baseline"},
        ],

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

        # Real-world experimental data — 5 real CASSI scenes from InverseNet
        "real_experimental": {
            "description": "5 real-world hyperspectral scenes captured with a CASSI prototype. Mismatch: dx=0.5, dy=0.3 px. Metrics are PSNR vs reference reconstruction.",
            "methods": [
                {
                    "method": "GAP-TV",
                    "calibrated_psnr": 21.69, "mismatched_psnr": 21.84, "residual_ratio": 1.8,
                },
                {
                    "method": "PnP-HSICNN",
                    "calibrated_psnr": 22.95, "mismatched_psnr": 23.41, "residual_ratio": 1.1,
                },
            ],
            "per_scene": [
                {"scene": "Scene 1", "gap_tv": {"cal_psnr": 23.83, "cal_ssim": 0.145, "mis_psnr": 24.18, "mis_ssim": 0.183, "ratio": 2.0}, "pnp_hsicnn": {"cal_psnr": 25.56, "cal_ssim": 0.382, "mis_psnr": 26.09, "mis_ssim": 0.439, "ratio": 1.1}},
                {"scene": "Scene 2", "gap_tv": {"cal_psnr": 21.40, "cal_ssim": 0.142, "mis_psnr": 21.52, "mis_ssim": 0.162, "ratio": 1.6}, "pnp_hsicnn": {"cal_psnr": 23.24, "cal_ssim": 0.452, "mis_psnr": 23.47, "mis_ssim": 0.479, "ratio": 1.1}},
                {"scene": "Scene 3", "gap_tv": {"cal_psnr": 20.48, "cal_ssim": 0.117, "mis_psnr": 20.61, "mis_ssim": 0.130, "ratio": 1.6}, "pnp_hsicnn": {"cal_psnr": 21.56, "cal_ssim": 0.267, "mis_psnr": 22.79, "mis_ssim": 0.388, "ratio": 1.1}},
                {"scene": "Scene 4", "gap_tv": {"cal_psnr": 21.11, "cal_ssim": 0.119, "mis_psnr": 21.20, "mis_ssim": 0.130, "ratio": 2.0}, "pnp_hsicnn": {"cal_psnr": 22.49, "cal_ssim": 0.320, "mis_psnr": 22.83, "mis_ssim": 0.358, "ratio": 1.2}},
                {"scene": "Scene 5", "gap_tv": {"cal_psnr": 21.61, "cal_ssim": 0.100, "mis_psnr": 21.71, "mis_ssim": 0.126, "ratio": 1.8}, "pnp_hsicnn": {"cal_psnr": 21.92, "cal_ssim": 0.182, "mis_psnr": 21.88, "mis_ssim": 0.175, "ratio": 1.1}},
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  CACTI — InverseNet validated results (6 DAVIS videos, 4 algorithms)
    # ══════════════════════════════════════════════════════════════════════════

    "cacti": {
        "challenge": [
            {"rank": 1, "method": "ELP-Unfolding + gradient", "public_score": 0.754, "dev_score": 0.616, "hidden_score": 0.575, "overall_score": 0.648, "details": {"public": {"psnr": 28.01, "ssim": 0.865, "consistency": 0.92}, "dev": {"psnr": 23.13, "ssim": 0.648, "consistency": 0.86}, "hidden": {"psnr": 21.74, "ssim": 0.587, "consistency": 0.83}}, "source": "InverseNet baseline"},
            {"rank": 2, "method": "EfficientSCI + gradient",  "public_score": 0.739, "dev_score": 0.604, "hidden_score": 0.563, "overall_score": 0.635, "details": {"public": {"psnr": 26.12, "ssim": 0.865, "consistency": 0.92}, "dev": {"psnr": 21.72, "ssim": 0.646, "consistency": 0.86}, "hidden": {"psnr": 20.47, "ssim": 0.584, "consistency": 0.83}}, "source": "InverseNet baseline"},
            {"rank": 3, "method": "GAP-TV + gradient",        "public_score": 0.674, "dev_score": 0.560, "hidden_score": 0.527, "overall_score": 0.587, "details": {"public": {"psnr": 24.79, "ssim": 0.735, "consistency": 0.91}, "dev": {"psnr": 21.11, "ssim": 0.559, "consistency": 0.84}, "hidden": {"psnr": 20.09, "ssim": 0.510, "consistency": 0.81}}, "source": "InverseNet baseline"},
            {"rank": 4, "method": "PnP-FFDNet + gradient",    "public_score": 0.685, "dev_score": 0.550, "hidden_score": 0.508, "overall_score": 0.581, "details": {"public": {"psnr": 24.13, "ssim": 0.766, "consistency": 0.93}, "dev": {"psnr": 19.25, "ssim": 0.554, "consistency": 0.87}, "hidden": {"psnr": 17.85, "ssim": 0.494, "consistency": 0.84}}, "source": "InverseNet baseline"},
        ],

        # B2 leaderboard — Scenario I (Ideal: no mismatch, no noise)
        "b2": [
            {"rank": 1, "method": "EfficientSCI",  "psnr": 35.39, "ssim": 0.973, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "ELP-Unfolding", "psnr": 34.09, "ssim": 0.965, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-FFDNet",    "psnr": 29.28, "ssim": 0.890, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "GAP-TV",        "psnr": 26.75, "ssim": 0.848, "source": "InverseNet", "adopted": False},
        ],

        # B2 scenario table — full 3-scenario comparison from InverseNet
        "b2_scenario_table": {
            "description": "6 DAVIS videos (256\u00d7256\u00d78 frames, CR=8), 4 algorithms, 3 scenarios. Mismatch: mask shift (0.5/0.3 px), temporal jitter (clock_offset=0.05, duty_cycle=0.95), gain (1.02), noise (\u03c3=1.0).",
            "scenarios": [
                {"id": "I",   "name": "Ideal",                 "color": "green", "description": "No mismatch, no noise \u2014 theoretical upper bound"},
                {"id": "II",  "name": "Mismatch (uncorrected)", "color": "red",   "description": "6-parameter mismatch + Poisson-Gaussian noise, no correction"},
                {"id": "III", "name": "Oracle correction",      "color": "blue",  "description": "True mismatch parameters known \u2014 calibration upper bound"},
            ],
            "methods": [
                {
                    "method": "GAP-TV", "type": "Classical", "mask_aware": True, "params": "0",
                    "s1_psnr": 26.75, "s1_ssim": 0.848,
                    "s2_psnr": 15.81, "s2_ssim": 0.305,
                    "s3_psnr": 26.01, "s3_ssim": 0.794,
                },
                {
                    "method": "PnP-FFDNet", "type": "PnP", "mask_aware": True, "params": "0",
                    "s1_psnr": 29.28, "s1_ssim": 0.890,
                    "s2_psnr": 11.43, "s2_ssim": 0.216,
                    "s3_psnr": 25.39, "s3_ssim": 0.820,
                },
                {
                    "method": "ELP-Unfolding", "type": "Deep Unfolding", "mask_aware": True, "params": "1.6M",
                    "s1_psnr": 34.09, "s1_ssim": 0.965,
                    "s2_psnr": 15.47, "s2_ssim": 0.308,
                    "s3_psnr": 29.40, "s3_ssim": 0.927,
                },
                {
                    "method": "EfficientSCI", "type": "Deep Learning", "mask_aware": True, "params": "4.2M",
                    "s1_psnr": 35.39, "s1_ssim": 0.973,
                    "s2_psnr": 14.81, "s2_ssim": 0.303,
                    "s3_psnr": 27.38, "s3_ssim": 0.927,
                },
            ],
            "method_names": ["GAP-TV", "PnP-FFDNet", "ELP-Unfolding", "EfficientSCI"],
            "per_scene": [
                {"scene": "Kobe",    "psnr_values": [26.70, 18.97, 26.50,  30.02, 16.00, 29.20,  34.07, 18.31, 32.63,  35.55, 18.21, 32.43]},
                {"scene": "Traffic", "psnr_values": [20.73, 13.99, 20.57,  24.06,  9.95, 23.37,  31.33, 13.90, 29.04,  32.19, 13.30, 27.08]},
                {"scene": "Runner",  "psnr_values": [29.34, 17.70, 28.85,  32.88, 13.40, 31.18,  38.14, 17.00, 34.06,  39.28, 16.65, 31.15]},
                {"scene": "Drop",    "psnr_values": [34.22, 13.64, 31.43,  38.71,  7.55, 21.91,  40.08, 13.52, 25.12,  42.36, 11.63, 21.95]},
                {"scene": "Crash",   "psnr_values": [24.80, 14.77, 24.45,  24.81, 10.11, 23.32,  29.38, 14.82, 26.84,  30.62, 14.25, 25.07]},
                {"scene": "Aerial",  "psnr_values": [25.22, 16.76, 24.95,  24.56, 12.79, 23.77,  30.43, 16.14, 28.80,  31.24, 15.92, 27.18]},
            ],
        },

        # Real-world experimental data — 4 real CACTI scenes from InverseNet
        "real_experimental": {
            "description": "4 real-world video scenes (CR=10) captured with a CACTI prototype. Mismatch: dx=0.5, dy=0.3 px. Metrics are residual norms and TV energy (no ground truth available for real data).",
            "methods": ["GAP-TV", "PnP-FFDNet"],
            "per_scene": [
                {"scene": "Domino",        "gap_tv": {"cal_residual": 8.0e-06, "mis_residual": 8.5e-05, "cal_tv": 10372.3, "mis_tv": 11616.9, "ratio": 10.625}, "pnp_ffdnet": {"cal_residual": 0.001981, "mis_residual": 0.004027, "cal_tv": 9683.4, "mis_tv": 10777.2, "ratio": 2.033}},
                {"scene": "Hand",          "gap_tv": {"cal_residual": 7.0e-06, "mis_residual": 7.7e-05, "cal_tv": 12550.8, "mis_tv": 23245.0, "ratio": 11.0},   "pnp_ffdnet": {"cal_residual": 0.002493, "mis_residual": 0.007049, "cal_tv": 7604.7, "mis_tv": 12232.8, "ratio": 2.828}},
                {"scene": "PendulumBall",  "gap_tv": {"cal_residual": 3.7e-05, "mis_residual": 0.000346, "cal_tv": 9687.5, "mis_tv": 9484.7, "ratio": 9.351},  "pnp_ffdnet": {"cal_residual": 0.009192, "mis_residual": 0.011502, "cal_tv": 7770.4, "mis_tv": 8005.5, "ratio": 1.251}},
                {"scene": "WaterBalloon",  "gap_tv": {"cal_residual": 1.4e-05, "mis_residual": 0.000147, "cal_tv": 10017.2, "mis_tv": 11161.1, "ratio": 10.5}, "pnp_ffdnet": {"cal_residual": 0.002637, "mis_residual": 0.004932, "cal_tv": 8292.5, "mis_tv": 8792.1, "ratio": 1.87}},
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  SPC — Block — InverseNet validated results (11 Set11 images, 4 algorithms)
    # ══════════════════════════════════════════════════════════════════════════

    "spc_block": {
        "challenge": [
            {"rank": 1, "method": "HATNet + gradient",     "public_score": 0.710, "dev_score": 0.630, "hidden_score": 0.612, "overall_score": 0.651, "details": {"public": {"psnr": 27.91, "ssim": 0.778, "consistency": 0.88}, "dev": {"psnr": 23.34, "ssim": 0.708, "consistency": 0.80}, "hidden": {"psnr": 22.51, "ssim": 0.696, "consistency": 0.77}}, "source": "InverseNet baseline"},
            {"rank": 2, "method": "ISTA-Net + gradient",   "public_score": 0.694, "dev_score": 0.634, "hidden_score": 0.614, "overall_score": 0.647, "details": {"public": {"psnr": 26.61, "ssim": 0.742, "consistency": 0.92}, "dev": {"psnr": 23.66, "ssim": 0.681, "consistency": 0.86}, "hidden": {"psnr": 22.81, "ssim": 0.663, "consistency": 0.83}}, "source": "InverseNet baseline"},
            {"rank": 3, "method": "FISTA-TV + gradient",   "public_score": 0.680, "dev_score": 0.618, "hidden_score": 0.600, "overall_score": 0.633, "details": {"public": {"psnr": 25.29, "ssim": 0.738, "consistency": 0.91}, "dev": {"psnr": 22.51, "ssim": 0.676, "consistency": 0.84}, "hidden": {"psnr": 21.74, "ssim": 0.659, "consistency": 0.81}}, "source": "InverseNet baseline"},
            {"rank": 4, "method": "PnP-DRUNet + gradient", "public_score": 0.627, "dev_score": 0.560, "hidden_score": 0.537, "overall_score": 0.575, "details": {"public": {"psnr": 22.99, "ssim": 0.643, "consistency": 0.93}, "dev": {"psnr": 20.41, "ssim": 0.556, "consistency": 0.87}, "hidden": {"psnr": 19.68, "ssim": 0.530, "consistency": 0.84}}, "source": "InverseNet baseline"},
        ],

        # B2 leaderboard — Scenario I (Ideal: no mismatch, no noise)
        "b2": [
            {"rank": 1, "method": "ISTA-Net",   "psnr": 31.85, "ssim": 0.916, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HATNet",     "psnr": 30.98, "ssim": 0.847, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-DRUNet", "psnr": 30.53, "ssim": 0.899, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "FISTA-TV",   "psnr": 28.06, "ssim": 0.852, "source": "InverseNet", "adopted": False},
        ],

        # B2 scenario table — full 3-scenario comparison from InverseNet
        "b2_scenario_table": {
            "description": "11 Set11 images (33\u00d733), 4 algorithms, 3 scenarios. Sampling rate: 15%. Mismatch: gain \u03b1=0.0015, measurement noise \u03c3=0.03.",
            "scenarios": [
                {"id": "I",   "name": "Ideal",                 "color": "green", "description": "No mismatch, no noise \u2014 theoretical upper bound"},
                {"id": "II",  "name": "Mismatch (uncorrected)", "color": "red",   "description": "Gain perturbation + measurement noise, no correction"},
                {"id": "III", "name": "Oracle correction",      "color": "blue",  "description": "True mismatch parameters known \u2014 calibration upper bound"},
            ],
            "methods": [
                {
                    "method": "FISTA-TV", "type": "Classical", "mask_aware": True, "params": "0",
                    "s1_psnr": 28.06, "s1_ssim": 0.852,
                    "s2_psnr": 18.51, "s2_ssim": 0.586,
                    "s3_psnr": 26.21, "s3_ssim": 0.759,
                },
                {
                    "method": "PnP-DRUNet", "type": "PnP", "mask_aware": True, "params": "0",
                    "s1_psnr": 30.53, "s1_ssim": 0.899,
                    "s2_psnr": 16.29, "s2_ssim": 0.415,
                    "s3_psnr": 23.65, "s3_ssim": 0.666,
                },
                {
                    "method": "HATNet", "type": "Deep Learning", "mask_aware": False, "params": "0.8M",
                    "s1_psnr": 30.98, "s1_ssim": 0.847,
                    "s2_psnr": 19.40, "s2_ssim": 0.648,
                    "s3_psnr": 29.78, "s3_ssim": 0.807,
                },
                {
                    "method": "ISTA-Net", "type": "Deep Unfolding", "mask_aware": True, "params": "0.3M",
                    "s1_psnr": 31.85, "s1_ssim": 0.916,
                    "s2_psnr": 19.02, "s2_ssim": 0.584,
                    "s3_psnr": 27.45, "s3_ssim": 0.760,
                },
            ],
            "method_names": ["FISTA-TV", "PnP-DRUNet", "HATNet", "ISTA-Net"],
            "per_scene": [
                {"scene": "Monarch",     "psnr_values": [28.04, 19.22, 26.15,  31.43, 17.25, 23.77,  30.91, 20.36, 29.77,  32.54, 19.90, 27.70]},
                {"scene": "Parrots",     "psnr_values": [27.40, 18.53, 26.18,  29.85, 16.18, 23.15,  31.63, 19.51, 30.42,  31.42, 18.99, 27.82]},
                {"scene": "Barbara",     "psnr_values": [24.48, 18.49, 23.79,  28.14, 16.26, 22.22,  30.72, 19.55, 29.45,  27.84, 19.02, 25.53]},
                {"scene": "Boats",       "psnr_values": [28.79, 18.86, 26.85,  31.57, 16.47, 24.25,  31.04, 19.58, 29.76,  32.91, 19.26, 27.89]},
                {"scene": "Cameraman",   "psnr_values": [26.04, 18.78, 24.96,  26.48, 16.69, 22.49,  29.68, 19.77, 28.68,  28.61, 19.31, 26.16]},
                {"scene": "Fingerprint", "psnr_values": [23.08, 17.33, 22.58,  26.36, 14.75, 21.05,  30.11, 18.84, 29.15,  28.10, 18.16, 25.56]},
                {"scene": "Flinstones",  "psnr_values": [24.64, 17.18, 23.59,  29.25, 15.33, 23.16,  29.48, 18.49, 28.52,  29.37, 18.01, 26.39]},
                {"scene": "Foreman",     "psnr_values": [35.10, 17.96, 30.42,  36.97, 15.53, 26.11,  32.80, 18.34, 31.31,  38.23, 18.20, 29.68]},
                {"scene": "House",       "psnr_values": [32.20, 18.90, 29.20,  35.52, 16.71, 25.97,  32.17, 19.34, 30.80,  35.70, 19.23, 29.21]},
                {"scene": "Lena",        "psnr_values": [28.97, 19.25, 27.13,  27.48, 16.97, 23.33,  31.22, 19.96, 29.89,  32.30, 19.67, 27.96]},
                {"scene": "Peppers",     "psnr_values": [29.95, 19.11, 27.51,  32.74, 17.00, 24.68,  31.05, 19.68, 29.87,  33.33, 19.49, 28.01]},
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  SPC — Kronecker (same Set11 baselines, different measurement matrix)
    # ══════════════════════════════════════════════════════════════════════════

    "spc_kronecker": {
        "challenge": [
            {"rank": 1, "method": "HATNet + gradient",     "public_score": 0.710, "dev_score": 0.630, "hidden_score": 0.612, "overall_score": 0.651, "details": {"public": {"psnr": 27.91, "ssim": 0.778, "consistency": 0.88}, "dev": {"psnr": 23.34, "ssim": 0.708, "consistency": 0.80}, "hidden": {"psnr": 22.51, "ssim": 0.696, "consistency": 0.77}}, "source": "InverseNet baseline"},
            {"rank": 2, "method": "ISTA-Net + gradient",   "public_score": 0.694, "dev_score": 0.634, "hidden_score": 0.614, "overall_score": 0.647, "details": {"public": {"psnr": 26.61, "ssim": 0.742, "consistency": 0.92}, "dev": {"psnr": 23.66, "ssim": 0.681, "consistency": 0.86}, "hidden": {"psnr": 22.81, "ssim": 0.663, "consistency": 0.83}}, "source": "InverseNet baseline"},
            {"rank": 3, "method": "FISTA-TV + gradient",   "public_score": 0.680, "dev_score": 0.618, "hidden_score": 0.600, "overall_score": 0.633, "details": {"public": {"psnr": 25.29, "ssim": 0.738, "consistency": 0.91}, "dev": {"psnr": 22.51, "ssim": 0.676, "consistency": 0.84}, "hidden": {"psnr": 21.74, "ssim": 0.659, "consistency": 0.81}}, "source": "InverseNet baseline"},
            {"rank": 4, "method": "PnP-DRUNet + gradient", "public_score": 0.627, "dev_score": 0.560, "hidden_score": 0.537, "overall_score": 0.575, "details": {"public": {"psnr": 22.99, "ssim": 0.643, "consistency": 0.93}, "dev": {"psnr": 20.41, "ssim": 0.556, "consistency": 0.87}, "hidden": {"psnr": 19.68, "ssim": 0.530, "consistency": 0.84}}, "source": "InverseNet baseline"},
        ],

        # B2 leaderboard — Scenario I (Ideal: no mismatch, no noise)
        "b2": [
            {"rank": 1, "method": "ISTA-Net",   "psnr": 31.85, "ssim": 0.916, "source": "InverseNet", "adopted": True},
            {"rank": 2, "method": "HATNet",     "psnr": 30.98, "ssim": 0.847, "source": "InverseNet", "adopted": False},
            {"rank": 3, "method": "PnP-DRUNet", "psnr": 30.53, "ssim": 0.899, "source": "InverseNet", "adopted": False},
            {"rank": 4, "method": "FISTA-TV",   "psnr": 28.06, "ssim": 0.852, "source": "InverseNet", "adopted": False},
        ],

        # B2 scenario table — full 3-scenario comparison from InverseNet
        "b2_scenario_table": {
            "description": "11 Set11 images (33\u00d733), 4 algorithms, 3 scenarios. Sampling rate: 15%. Mismatch: gain \u03b1=0.0015, measurement noise \u03c3=0.03.",
            "scenarios": [
                {"id": "I",   "name": "Ideal",                 "color": "green", "description": "No mismatch, no noise \u2014 theoretical upper bound"},
                {"id": "II",  "name": "Mismatch (uncorrected)", "color": "red",   "description": "Gain perturbation + measurement noise, no correction"},
                {"id": "III", "name": "Oracle correction",      "color": "blue",  "description": "True mismatch parameters known \u2014 calibration upper bound"},
            ],
            "methods": [
                {
                    "method": "FISTA-TV", "type": "Classical", "mask_aware": True, "params": "0",
                    "s1_psnr": 28.06, "s1_ssim": 0.852,
                    "s2_psnr": 18.51, "s2_ssim": 0.586,
                    "s3_psnr": 26.21, "s3_ssim": 0.759,
                },
                {
                    "method": "PnP-DRUNet", "type": "PnP", "mask_aware": True, "params": "0",
                    "s1_psnr": 30.53, "s1_ssim": 0.899,
                    "s2_psnr": 16.29, "s2_ssim": 0.415,
                    "s3_psnr": 23.65, "s3_ssim": 0.666,
                },
                {
                    "method": "HATNet", "type": "Deep Learning", "mask_aware": False, "params": "0.8M",
                    "s1_psnr": 30.98, "s1_ssim": 0.847,
                    "s2_psnr": 19.40, "s2_ssim": 0.648,
                    "s3_psnr": 29.78, "s3_ssim": 0.807,
                },
                {
                    "method": "ISTA-Net", "type": "Deep Unfolding", "mask_aware": True, "params": "0.3M",
                    "s1_psnr": 31.85, "s1_ssim": 0.916,
                    "s2_psnr": 19.02, "s2_ssim": 0.584,
                    "s3_psnr": 27.45, "s3_ssim": 0.760,
                },
            ],
            "method_names": ["FISTA-TV", "PnP-DRUNet", "HATNet", "ISTA-Net"],
            "per_scene": [
                {"scene": "Monarch",     "psnr_values": [28.04, 19.22, 26.15,  31.43, 17.25, 23.77,  30.91, 20.36, 29.77,  32.54, 19.90, 27.70]},
                {"scene": "Parrots",     "psnr_values": [27.40, 18.53, 26.18,  29.85, 16.18, 23.15,  31.63, 19.51, 30.42,  31.42, 18.99, 27.82]},
                {"scene": "Barbara",     "psnr_values": [24.48, 18.49, 23.79,  28.14, 16.26, 22.22,  30.72, 19.55, 29.45,  27.84, 19.02, 25.53]},
                {"scene": "Boats",       "psnr_values": [28.79, 18.86, 26.85,  31.57, 16.47, 24.25,  31.04, 19.58, 29.76,  32.91, 19.26, 27.89]},
                {"scene": "Cameraman",   "psnr_values": [26.04, 18.78, 24.96,  26.48, 16.69, 22.49,  29.68, 19.77, 28.68,  28.61, 19.31, 26.16]},
                {"scene": "Fingerprint", "psnr_values": [23.08, 17.33, 22.58,  26.36, 14.75, 21.05,  30.11, 18.84, 29.15,  28.10, 18.16, 25.56]},
                {"scene": "Flinstones",  "psnr_values": [24.64, 17.18, 23.59,  29.25, 15.33, 23.16,  29.48, 18.49, 28.52,  29.37, 18.01, 26.39]},
                {"scene": "Foreman",     "psnr_values": [35.10, 17.96, 30.42,  36.97, 15.53, 26.11,  32.80, 18.34, 31.31,  38.23, 18.20, 29.68]},
                {"scene": "House",       "psnr_values": [32.20, 18.90, 29.20,  35.52, 16.71, 25.97,  32.17, 19.34, 30.80,  35.70, 19.23, 29.21]},
                {"scene": "Lena",        "psnr_values": [28.97, 19.25, 27.13,  27.48, 16.97, 23.33,  31.22, 19.96, 29.89,  32.30, 19.67, 27.96]},
                {"scene": "Peppers",     "psnr_values": [29.95, 19.11, 27.51,  32.74, 17.00, 24.68,  31.05, 19.68, 29.87,  33.33, 19.49, 28.01]},
            ],
        },
    },
}
