#!/usr/bin/env python3
"""Auto-populate PWM-SyS Layer A system descriptors for all 168 imaging modalities.

Pulls from existing PWM data:
  - MODALITY_CATALOG:     carrier, category, display_name, mismatch_params
  - get_algorithms():     solver name, type, citation
  - CATEGORY_REAL_SCORES: best PSNR/SSIM per modality
  - get_score_key():      score routing

Hardware properties (shots, resolution, cost, operator skill) come from
curated lookup tables keyed by modality ID, with category-level defaults.

Output: benchmark-data/system_catalog.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add platform root to path so we can import benchmark_database
_PLATFORM_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PLATFORM_DIR.parent
sys.path.insert(0, str(_PLATFORM_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from pwm_platform.services.benchmark_database._modality_catalog import MODALITY_CATALOG
from pwm_platform.services.benchmark_database._algorithm_catalog import (
    get_algorithms,
    get_score_key,
    CATEGORY_REAL_SCORES,
)

# ---------------------------------------------------------------------------
# Per-modality hardware property overrides
# Fields: shots, max_fps, res_um, dims, capital_k, operator, solver_latency_s
# ---------------------------------------------------------------------------

_MODALITY_HARDWARE: dict[str, dict] = {
    # ── Compressive / Computational ───────────────────────────────────────
    "cacti":               {"shots": 1,     "max_fps": 1e8,   "res_um": 5,     "dims": "3D(x,y,t)",       "capital_k": 15,    "operator": "expert",      "solver_latency_s": 2.1},
    "cassi":               {"shots": 1,     "max_fps": 30,    "res_um": 5,     "dims": "3D(x,y,lam)",     "capital_k": 20,    "operator": "expert",      "solver_latency_s": 3.5},
    "spc":                 {"shots": 1000,  "max_fps": 1,     "res_um": 100,   "dims": "2D",              "capital_k": 2,     "operator": "technician",  "solver_latency_s": 0.5},
    "cup":                 {"shots": 1,     "max_fps": 1e10,  "res_um": 100,   "dims": "3D(x,y,t)",       "capital_k": 80,    "operator": "specialist",  "solver_latency_s": 30},
    "coded_exposure":      {"shots": 1,     "max_fps": 100,   "res_um": 5,     "dims": "2D+blur",         "capital_k": 3,     "operator": "technician",  "solver_latency_s": 0.01},
    "lensless":            {"shots": 1,     "max_fps": 30,    "res_um": 50,    "dims": "2D/3D",           "capital_k": 0.5,   "operator": "untrained",   "solver_latency_s": 0.1},
    "ghost_imaging":       {"shots": 1000,  "max_fps": 0.1,   "res_um": 100,   "dims": "2D",              "capital_k": 10,    "operator": "expert",      "solver_latency_s": 0.5},
    "entangled_photon":    {"shots": 1000,  "max_fps": 0.01,  "res_um": 50,    "dims": "2D",              "capital_k": 100,   "operator": "specialist",  "solver_latency_s": 5},
    "quantum_illumination":{"shots": 1000,  "max_fps": 0.001, "res_um": 1000,  "dims": "2D",              "capital_k": 200,   "operator": "specialist",  "solver_latency_s": 1},

    # ── Optical Microscopy ────────────────────────────────────────────────
    "widefield":           {"shots": 1,     "max_fps": 100,   "res_um": 0.3,   "dims": "2D",              "capital_k": 30,    "operator": "technician",  "solver_latency_s": 0.05},
    "widefield_lowdose":   {"shots": 1,     "max_fps": 100,   "res_um": 0.3,   "dims": "2D",              "capital_k": 30,    "operator": "technician",  "solver_latency_s": 0.05},
    "confocal_3d":         {"shots": 500,   "max_fps": 0.5,   "res_um": 0.2,   "dims": "3D(x,y,z)",       "capital_k": 200,   "operator": "expert",      "solver_latency_s": 5},
    "confocal_livecell":   {"shots": 50,    "max_fps": 5,     "res_um": 0.2,   "dims": "3D+t",            "capital_k": 200,   "operator": "expert",      "solver_latency_s": 0.5},
    "confocal_endomicroscopy": {"shots": 1, "max_fps": 30,    "res_um": 1,     "dims": "2D",              "capital_k": 100,   "operator": "technician",  "solver_latency_s": 0.1},
    "spinning_disk":       {"shots": 100,   "max_fps": 10,    "res_um": 0.25,  "dims": "3D(x,y,z)",       "capital_k": 250,   "operator": "expert",      "solver_latency_s": 1},
    "two_photon":          {"shots": 500,   "max_fps": 1,     "res_um": 0.3,   "dims": "3D(x,y,z)",       "capital_k": 400,   "operator": "expert",      "solver_latency_s": 2},
    "three_photon":        {"shots": 500,   "max_fps": 0.5,   "res_um": 0.3,   "dims": "3D(x,y,z)",       "capital_k": 600,   "operator": "specialist",  "solver_latency_s": 5},
    "lightsheet":          {"shots": 200,   "max_fps": 5,     "res_um": 0.4,   "dims": "3D(x,y,z)",       "capital_k": 150,   "operator": "expert",      "solver_latency_s": 2},
    "lattice_lightsheet":  {"shots": 200,   "max_fps": 10,    "res_um": 0.2,   "dims": "3D(x,y,z)+t",     "capital_k": 500,   "operator": "specialist",  "solver_latency_s": 2},
    "sim":                 {"shots": 9,     "max_fps": 10,    "res_um": 0.1,   "dims": "2D/3D",           "capital_k": 200,   "operator": "expert",      "solver_latency_s": 0.5},
    "palm_storm":          {"shots": 10000, "max_fps": 0.001, "res_um": 0.02,  "dims": "2D/3D",           "capital_k": 150,   "operator": "expert",      "solver_latency_s": 30},
    "sted":                {"shots": 500,   "max_fps": 1,     "res_um": 0.05,  "dims": "2D/3D",           "capital_k": 500,   "operator": "specialist",  "solver_latency_s": 2},
    "minflux":             {"shots": 10000, "max_fps": 0.001, "res_um": 0.002, "dims": "3D",              "capital_k": 800,   "operator": "specialist",  "solver_latency_s": 60},
    "dna_paint":           {"shots": 50000, "max_fps": 1e-4,  "res_um": 0.01,  "dims": "2D/3D",           "capital_k": 150,   "operator": "specialist",  "solver_latency_s": 120},
    "expansion":           {"shots": 1,     "max_fps": 1,     "res_um": 0.07,  "dims": "3D",              "capital_k": 50,    "operator": "expert",      "solver_latency_s": 10},
    "tirf":                {"shots": 1,     "max_fps": 100,   "res_um": 0.2,   "dims": "2D(surface)",     "capital_k": 100,   "operator": "expert",      "solver_latency_s": 1},
    "ism":                 {"shots": 100,   "max_fps": 5,     "res_um": 0.15,  "dims": "2D",              "capital_k": 200,   "operator": "expert",      "solver_latency_s": 1},
    "flim":                {"shots": 100,   "max_fps": 0.1,   "res_um": 0.3,   "dims": "2D+tau",          "capital_k": 150,   "operator": "expert",      "solver_latency_s": 2},
    "shg":                 {"shots": 500,   "max_fps": 1,     "res_um": 0.3,   "dims": "2D",              "capital_k": 300,   "operator": "expert",      "solver_latency_s": 1},
    "srs":                 {"shots": 500,   "max_fps": 1,     "res_um": 0.3,   "dims": "2D+lam",          "capital_k": 300,   "operator": "expert",      "solver_latency_s": 1},

    # ── Coherent / Phase ──────────────────────────────────────────────────
    "holography":          {"shots": 1,     "max_fps": 100,   "res_um": 0.3,   "dims": "2D+phase",        "capital_k": 50,    "operator": "expert",      "solver_latency_s": 0.1},
    "phase_contrast":      {"shots": 1,     "max_fps": 100,   "res_um": 0.3,   "dims": "2D+phase",        "capital_k": 60,    "operator": "expert",      "solver_latency_s": 0.1},
    "dic":                 {"shots": 1,     "max_fps": 100,   "res_um": 0.2,   "dims": "2D+gradient",     "capital_k": 80,    "operator": "technician",  "solver_latency_s": 0.1},
    "odt":                 {"shots": 50,    "max_fps": 1,     "res_um": 0.2,   "dims": "3D(RI)",          "capital_k": 100,   "operator": "specialist",  "solver_latency_s": 5},
    "phase_retrieval":     {"shots": 3,     "max_fps": 30,    "res_um": 0.3,   "dims": "2D+phase",        "capital_k": 30,    "operator": "expert",      "solver_latency_s": 1},
    "ptychography":        {"shots": 100,   "max_fps": 0.1,   "res_um": 0.01,  "dims": "2D+phase",        "capital_k": 200,   "operator": "specialist",  "solver_latency_s": 10},
    "fpm":                 {"shots": 100,   "max_fps": 0.5,   "res_um": 0.1,   "dims": "2D+phase",        "capital_k": 20,    "operator": "expert",      "solver_latency_s": 5},
    "dark_field":          {"shots": 1,     "max_fps": 30,    "res_um": 0.5,   "dims": "2D",              "capital_k": 50,    "operator": "technician",  "solver_latency_s": 0.05},
    "polarization":        {"shots": 1,     "max_fps": 30,    "res_um": 5,     "dims": "2D+Stokes",       "capital_k": 20,    "operator": "technician",  "solver_latency_s": 0.5},
    "talbot_lau":          {"shots": 4,     "max_fps": 5,     "res_um": 10,    "dims": "2D(abs,dpc,df)",  "capital_k": 100,   "operator": "expert",      "solver_latency_s": 0.01},
    "integral":            {"shots": 1,     "max_fps": 10,    "res_um": 10,    "dims": "3D(x,y,lam)",     "capital_k": 200,   "operator": "specialist",  "solver_latency_s": 2},
    "matrix":              {"shots": 100,   "max_fps": 0.1,   "res_um": 0.3,   "dims": "3D(x,y,z)",       "capital_k": 100,   "operator": "specialist",  "solver_latency_s": 10},

    # ── Medical X-ray / CT ────────────────────────────────────────────────
    "ct":                  {"shots": 1000,  "max_fps": 0.5,   "res_um": 300,   "dims": "3D(x,y,z)",       "capital_k": 1000,  "operator": "technician",  "solver_latency_s": 5},
    "cbct":                {"shots": 300,   "max_fps": 0.2,   "res_um": 200,   "dims": "3D(x,y,z)",       "capital_k": 200,   "operator": "technician",  "solver_latency_s": 5},
    "spectral_ct":         {"shots": 1000,  "max_fps": 0.3,   "res_um": 300,   "dims": "3D+E",            "capital_k": 2000,  "operator": "technician",  "solver_latency_s": 10},
    "industrial_ct":       {"shots": 1000,  "max_fps": 0.01,  "res_um": 5,     "dims": "3D(x,y,z)",       "capital_k": 300,   "operator": "expert",      "solver_latency_s": 30},
    "xray_radiography":    {"shots": 1,     "max_fps": 30,    "res_um": 100,   "dims": "2D",              "capital_k": 50,    "operator": "technician",  "solver_latency_s": 0.05},
    "mammography":         {"shots": 1,     "max_fps": 2,     "res_um": 70,    "dims": "2D",              "capital_k": 200,   "operator": "technician",  "solver_latency_s": 0.1},
    "digital_breast_tomo": {"shots": 15,    "max_fps": 1,     "res_um": 100,   "dims": "3D",              "capital_k": 400,   "operator": "technician",  "solver_latency_s": 5},
    "fluoroscopy":         {"shots": 1,     "max_fps": 30,    "res_um": 200,   "dims": "2D(t)",           "capital_k": 300,   "operator": "technician",  "solver_latency_s": 0.03},
    "angiography":         {"shots": 10,    "max_fps": 15,    "res_um": 200,   "dims": "2D(t)",           "capital_k": 500,   "operator": "technician",  "solver_latency_s": 0.1},
    "dexa":                {"shots": 1,     "max_fps": 1,     "res_um": 500,   "dims": "2D(dual-E)",      "capital_k": 50,    "operator": "technician",  "solver_latency_s": 0.1},

    # ── Medical MRI ───────────────────────────────────────────────────────
    "MRI":                 {"shots": 100,   "max_fps": 0.1,   "res_um": 500,   "dims": "3D+contrast",     "capital_k": 2000,  "operator": "technician",  "solver_latency_s": 2},
    "mri":                 {"shots": 100,   "max_fps": 0.1,   "res_um": 500,   "dims": "3D+contrast",     "capital_k": 2000,  "operator": "technician",  "solver_latency_s": 2},
    "asl_mri":             {"shots": 100,   "max_fps": 0.05,  "res_um": 2000,  "dims": "2D+perfusion",    "capital_k": 2000,  "operator": "expert",      "solver_latency_s": 5},
    "cest_mri":            {"shots": 100,   "max_fps": 0.02,  "res_um": 1000,  "dims": "2D+Z-spec",       "capital_k": 2000,  "operator": "specialist",  "solver_latency_s": 10},
    "diffusion_mri":       {"shots": 100,   "max_fps": 0.05,  "res_um": 1500,  "dims": "3D+diffusion",    "capital_k": 2000,  "operator": "expert",      "solver_latency_s": 10},
    "fmri":                {"shots": 100,   "max_fps": 0.5,   "res_um": 2000,  "dims": "3D+t(BOLD)",      "capital_k": 2000,  "operator": "expert",      "solver_latency_s": 2},
    "mr_elastography":     {"shots": 100,   "max_fps": 0.1,   "res_um": 2000,  "dims": "3D+stiffness",    "capital_k": 2000,  "operator": "specialist",  "solver_latency_s": 10},
    "mr_fingerprinting":   {"shots": 1000,  "max_fps": 0.01,  "res_um": 1000,  "dims": "2D+multi-param",  "capital_k": 2000,  "operator": "specialist",  "solver_latency_s": 5},
    "mra":                 {"shots": 100,   "max_fps": 0.1,   "res_um": 500,   "dims": "3D(vessels)",      "capital_k": 2000,  "operator": "technician",  "solver_latency_s": 5},
    "mrs":                 {"shots": 100,   "max_fps": 0.01,  "res_um": 10000, "dims": "1D+chem",         "capital_k": 2000,  "operator": "specialist",  "solver_latency_s": 2},
    "swi":                 {"shots": 100,   "max_fps": 0.1,   "res_um": 500,   "dims": "3D+suscept",      "capital_k": 2000,  "operator": "expert",      "solver_latency_s": 5},

    # ── Medical Ultrasound ────────────────────────────────────────────────
    "ultrasound":          {"shots": 1,     "max_fps": 100,   "res_um": 300,   "dims": "2D",              "capital_k": 30,    "operator": "technician",  "solver_latency_s": 0.001},
    "doppler_ultrasound":  {"shots": 1,     "max_fps": 50,    "res_um": 300,   "dims": "2D+velocity",     "capital_k": 50,    "operator": "technician",  "solver_latency_s": 0.005},
    "elastography":        {"shots": 2,     "max_fps": 20,    "res_um": 500,   "dims": "2D+stiffness",    "capital_k": 80,    "operator": "technician",  "solver_latency_s": 0.1},
    "ceus":                {"shots": 1,     "max_fps": 30,    "res_um": 300,   "dims": "2D+perfusion",    "capital_k": 80,    "operator": "expert",      "solver_latency_s": 0.05},
    "ivus":                {"shots": 1,     "max_fps": 30,    "res_um": 100,   "dims": "2D(cross-sec)",   "capital_k": 200,   "operator": "specialist",  "solver_latency_s": 0.01},
    "us_mri":              {"shots": 1,     "max_fps": 10,    "res_um": 500,   "dims": "3D(fused)",       "capital_k": 100,   "operator": "expert",      "solver_latency_s": 1},

    # ── Medical Nuclear ───────────────────────────────────────────────────
    "pet":                 {"shots": 1e7,   "max_fps": 0.001, "res_um": 4000,  "dims": "3D",              "capital_k": 2000,  "operator": "technician",  "solver_latency_s": 30},
    "pet_ct":              {"shots": 1e7,   "max_fps": 0.001, "res_um": 2000,  "dims": "3D+anat",         "capital_k": 3000,  "operator": "technician",  "solver_latency_s": 30},
    "pet_mr":              {"shots": 1e7,   "max_fps": 0.001, "res_um": 2000,  "dims": "3D+multi",        "capital_k": 5000,  "operator": "specialist",  "solver_latency_s": 60},
    "spect":               {"shots": 1e7,   "max_fps": 0.001, "res_um": 8000,  "dims": "3D",              "capital_k": 500,   "operator": "technician",  "solver_latency_s": 30},
    "spect_ct":            {"shots": 1e7,   "max_fps": 0.001, "res_um": 5000,  "dims": "3D+anat",         "capital_k": 1000,  "operator": "technician",  "solver_latency_s": 30},

    # ── Medical Optical ───────────────────────────────────────────────────
    "oct":                 {"shots": 1,     "max_fps": 1e5,   "res_um": 5,     "dims": "3D(x,y,z)",       "capital_k": 80,    "operator": "technician",  "solver_latency_s": 1},
    "octa":                {"shots": 2,     "max_fps": 5e4,   "res_um": 10,    "dims": "3D(vessels)",      "capital_k": 100,   "operator": "technician",  "solver_latency_s": 2},
    "fundus":              {"shots": 1,     "max_fps": 10,    "res_um": 10,    "dims": "2D",              "capital_k": 20,    "operator": "technician",  "solver_latency_s": 0.05},
    "endoscopy":           {"shots": 1,     "max_fps": 30,    "res_um": 50,    "dims": "2D",              "capital_k": 30,    "operator": "technician",  "solver_latency_s": 0.03},
    "photoacoustic":       {"shots": 100,   "max_fps": 5,     "res_um": 50,    "dims": "3D",              "capital_k": 200,   "operator": "expert",      "solver_latency_s": 5},
    "dot":                 {"shots": 100,   "max_fps": 1,     "res_um": 5000,  "dims": "3D",              "capital_k": 100,   "operator": "expert",      "solver_latency_s": 10},
    "nirs_brain":          {"shots": 1,     "max_fps": 10,    "res_um": 10000, "dims": "2D(cortex)",      "capital_k": 30,    "operator": "technician",  "solver_latency_s": 1},
    "bioluminescence_tomo":{"shots": 1,     "max_fps": 0.1,   "res_um": 1000,  "dims": "3D",              "capital_k": 100,   "operator": "specialist",  "solver_latency_s": 30},

    # ── Radiotherapy ──────────────────────────────────────────────────────
    "brachytherapy_img":   {"shots": 1,     "max_fps": 1,     "res_um": 500,   "dims": "3D",              "capital_k": 500,   "operator": "specialist",  "solver_latency_s": 5},
    "portal_imaging":      {"shots": 1,     "max_fps": 15,    "res_um": 400,   "dims": "2D",              "capital_k": 200,   "operator": "technician",  "solver_latency_s": 0.1},
    "proton_radiography":  {"shots": 10000, "max_fps": 0.01,  "res_um": 500,   "dims": "2D(RSP)",         "capital_k": 10000, "operator": "specialist",  "solver_latency_s": 30},
    "proton_therapy_img":  {"shots": 100,   "max_fps": 0.1,   "res_um": 1000,  "dims": "3D",              "capital_k": 10000, "operator": "specialist",  "solver_latency_s": 10},
    "ct_fluorescence":     {"shots": 1000,  "max_fps": 0.01,  "res_um": 200,   "dims": "3D+element",      "capital_k": 1000,  "operator": "specialist",  "solver_latency_s": 30},
    "magnetic_particle":   {"shots": 100,   "max_fps": 40,    "res_um": 1000,  "dims": "3D",              "capital_k": 500,   "operator": "expert",      "solver_latency_s": 2},

    # ── Electron Microscopy ───────────────────────────────────────────────
    "sem":                 {"shots": 10000, "max_fps": 0.01,  "res_um": 0.001, "dims": "2D",              "capital_k": 200,   "operator": "expert",      "solver_latency_s": 1},
    "tem":                 {"shots": 1,     "max_fps": 10,    "res_um": 5e-5,  "dims": "2D",              "capital_k": 3000,  "operator": "specialist",  "solver_latency_s": 5},
    "stem":                {"shots": 10000, "max_fps": 0.001, "res_um": 8e-5,  "dims": "2D",              "capital_k": 2000,  "operator": "specialist",  "solver_latency_s": 2},
    "cryo_em":             {"shots": 1e5,   "max_fps": 1e-4,  "res_um": 3e-4,  "dims": "3D",              "capital_k": 3000,  "operator": "specialist",  "solver_latency_s": 3600},
    "cryo_et":             {"shots": 60,    "max_fps": 0.001, "res_um": 0.002, "dims": "3D",              "capital_k": 3000,  "operator": "specialist",  "solver_latency_s": 1800},
    "eels":                {"shots": 100,   "max_fps": 0.01,  "res_um": 1e-4,  "dims": "1D+spec",         "capital_k": 2000,  "operator": "specialist",  "solver_latency_s": 5},
    "electron_diffraction":{"shots": 1,     "max_fps": 1,     "res_um": 1e-4,  "dims": "2D(recip)",       "capital_k": 2000,  "operator": "specialist",  "solver_latency_s": 10},
    "electron_holography": {"shots": 1,     "max_fps": 1,     "res_um": 5e-4,  "dims": "2D+phase",        "capital_k": 2000,  "operator": "specialist",  "solver_latency_s": 5},
    "electron_tomography": {"shots": 100,   "max_fps": 0.001, "res_um": 0.001, "dims": "3D",              "capital_k": 2000,  "operator": "specialist",  "solver_latency_s": 600},

    # ── Ion / Mass Spec ───────────────────────────────────────────────────
    "atom_probe":          {"shots": 1e7,   "max_fps": 1e-4,  "res_um": 3e-4,  "dims": "3D+chem",         "capital_k": 2000,  "operator": "specialist",  "solver_latency_s": 300},
    "sims":                {"shots": 10000, "max_fps": 0.01,  "res_um": 0.05,  "dims": "2D+mass",         "capital_k": 1000,  "operator": "specialist",  "solver_latency_s": 10},
    "maldi_msi":           {"shots": 10000, "max_fps": 0.01,  "res_um": 10,    "dims": "2D+mass",         "capital_k": 500,   "operator": "specialist",  "solver_latency_s": 10},
    "desi":                {"shots": 10000, "max_fps": 0.1,   "res_um": 100,   "dims": "2D+mass",         "capital_k": 300,   "operator": "expert",      "solver_latency_s": 5},
    "libs":                {"shots": 10000, "max_fps": 10,    "res_um": 50,    "dims": "2D+element",      "capital_k": 100,   "operator": "expert",      "solver_latency_s": 1},

    # ── Scanning Probe ────────────────────────────────────────────────────
    "afm":                 {"shots": 1e5,   "max_fps": 0.001, "res_um": 1e-4,  "dims": "2D(topo)",        "capital_k": 100,   "operator": "expert",      "solver_latency_s": 5},
    "stm":                 {"shots": 1e5,   "max_fps": 1e-4,  "res_um": 1e-5,  "dims": "2D(LDOS)",        "capital_k": 200,   "operator": "specialist",  "solver_latency_s": 5},
    "nsom":                {"shots": 1e5,   "max_fps": 0.001, "res_um": 0.05,  "dims": "2D+optical",      "capital_k": 150,   "operator": "specialist",  "solver_latency_s": 5},
    "mfm":                 {"shots": 1e5,   "max_fps": 0.001, "res_um": 0.03,  "dims": "2D(magnetic)",    "capital_k": 120,   "operator": "expert",      "solver_latency_s": 5},
    "ebsd":                {"shots": 10000, "max_fps": 0.1,   "res_um": 0.05,  "dims": "2D(orient)",      "capital_k": 200,   "operator": "expert",      "solver_latency_s": 5},
    "fib_sem":             {"shots": 1000,  "max_fps": 1e-4,  "res_um": 0.005, "dims": "3D",              "capital_k": 1000,  "operator": "specialist",  "solver_latency_s": 600},

    # ── NDT / Industrial ──────────────────────────────────────────────────
    "active_thermography": {"shots": 1,     "max_fps": 100,   "res_um": 500,   "dims": "2D+t(IR)",        "capital_k": 30,    "operator": "technician",  "solver_latency_s": 2},
    "acoustic_microscopy": {"shots": 1e5,   "max_fps": 0.01,  "res_um": 1,     "dims": "2D",              "capital_k": 80,    "operator": "expert",      "solver_latency_s": 2},
    "acoustic_emission":   {"shots": 1,     "max_fps": 1e6,   "res_um": 5000,  "dims": "2D(source)",      "capital_k": 10,    "operator": "technician",  "solver_latency_s": 5},
    "eddy_current":        {"shots": 100,   "max_fps": 10,    "res_um": 500,   "dims": "2D",              "capital_k": 15,    "operator": "technician",  "solver_latency_s": 0.1},
    "xray_ndt":            {"shots": 1,     "max_fps": 10,    "res_um": 100,   "dims": "2D",              "capital_k": 50,    "operator": "technician",  "solver_latency_s": 0.1},
    "shearography":        {"shots": 1,     "max_fps": 30,    "res_um": 100,   "dims": "2D(strain)",      "capital_k": 40,    "operator": "technician",  "solver_latency_s": 0.5},
    "ultrasonic_phased_array": {"shots": 1, "max_fps": 100,   "res_um": 300,   "dims": "2D/3D",           "capital_k": 30,    "operator": "technician",  "solver_latency_s": 0.5},

    # ── Spectroscopy / Hyperspectral ──────────────────────────────────────
    "raman_imaging":       {"shots": 10000, "max_fps": 0.01,  "res_um": 0.5,   "dims": "2D+spec",         "capital_k": 200,   "operator": "expert",      "solver_latency_s": 10},
    "cars":                {"shots": 1,     "max_fps": 30,    "res_um": 0.3,   "dims": "2D+spec",         "capital_k": 300,   "operator": "expert",      "solver_latency_s": 0.5},
    "brillouin":           {"shots": 10000, "max_fps": 0.01,  "res_um": 0.3,   "dims": "2D+mech",         "capital_k": 200,   "operator": "specialist",  "solver_latency_s": 10},
    "ftir_imaging":        {"shots": 100,   "max_fps": 0.1,   "res_um": 5,     "dims": "2D+IR-spec",      "capital_k": 100,   "operator": "expert",      "solver_latency_s": 5},
    "cathodoluminescence": {"shots": 10000, "max_fps": 0.01,  "res_um": 0.01,  "dims": "2D+spec",         "capital_k": 500,   "operator": "specialist",  "solver_latency_s": 5},
    "edx_mapping":         {"shots": 10000, "max_fps": 0.01,  "res_um": 0.01,  "dims": "2D+element",      "capital_k": 300,   "operator": "expert",      "solver_latency_s": 5},
    "xrf_imaging":         {"shots": 10000, "max_fps": 0.1,   "res_um": 20,    "dims": "2D+element",      "capital_k": 100,   "operator": "expert",      "solver_latency_s": 2},
    "xrf_tomo":            {"shots": 1e5,   "max_fps": 0.001, "res_um": 50,    "dims": "3D+element",      "capital_k": 500,   "operator": "specialist",  "solver_latency_s": 60},

    # ── Remote Sensing ────────────────────────────────────────────────────
    "sar":                 {"shots": 1,     "max_fps": 0.01,  "res_um": 1e6,   "dims": "2D(complex)",     "capital_k": 10000, "operator": "specialist",  "solver_latency_s": 10},
    "polsar":              {"shots": 1,     "max_fps": 0.01,  "res_um": 1e6,   "dims": "2D+pol",          "capital_k": 15000, "operator": "specialist",  "solver_latency_s": 15},
    "insar":               {"shots": 2,     "max_fps": 0.001, "res_um": 1e6,   "dims": "2D(deform)",      "capital_k": 15000, "operator": "specialist",  "solver_latency_s": 30},
    "lidar":               {"shots": 1e5,   "max_fps": 10,    "res_um": 10000, "dims": "3D(point)",       "capital_k": 50,    "operator": "technician",  "solver_latency_s": 0.5},
    "flash_lidar":         {"shots": 1,     "max_fps": 30,    "res_um": 10000, "dims": "3D(depth)",       "capital_k": 20,    "operator": "untrained",   "solver_latency_s": 0.05},
    "hyperspectral_remote":{"shots": 1,     "max_fps": 0.1,   "res_um": 5e5,   "dims": "2D+lam",          "capital_k": 500,   "operator": "expert",      "solver_latency_s": 5},
    "multispectral_sat":   {"shots": 1,     "max_fps": 0.01,  "res_um": 1e6,   "dims": "2D+bands",        "capital_k": 5000,  "operator": "specialist",  "solver_latency_s": 2},
    "ocean_color":         {"shots": 1,     "max_fps": 0.01,  "res_um": 1e7,   "dims": "2D+bands",        "capital_k": 3000,  "operator": "specialist",  "solver_latency_s": 2},
    "passive_microwave":   {"shots": 1,     "max_fps": 0.01,  "res_um": 1e8,   "dims": "2D+freq",         "capital_k": 5000,  "operator": "specialist",  "solver_latency_s": 2},

    # ── Geophysics ────────────────────────────────────────────────────────
    "fwi":                 {"shots": 100,   "max_fps": 1e-4,  "res_um": 1e7,   "dims": "3D(velocity)",    "capital_k": 5000,  "operator": "specialist",  "solver_latency_s": 3600},
    "seismic_tomo":        {"shots": 100,   "max_fps": 1e-4,  "res_um": 1e8,   "dims": "3D(velocity)",    "capital_k": 10000, "operator": "specialist",  "solver_latency_s": 3600},
    "gpr":                 {"shots": 100,   "max_fps": 1,     "res_um": 1e5,   "dims": "2D/3D",           "capital_k": 20,    "operator": "technician",  "solver_latency_s": 5},
    "impedance_tomo":      {"shots": 100,   "max_fps": 10,    "res_um": 5e4,   "dims": "2D",              "capital_k": 10,    "operator": "technician",  "solver_latency_s": 1},
    "muon_tomo":           {"shots": 1e7,   "max_fps": 1e-4,  "res_um": 1e6,   "dims": "3D",              "capital_k": 500,   "operator": "specialist",  "solver_latency_s": 3600},

    # ── Astronomy ─────────────────────────────────────────────────────────
    "adaptive_optics":     {"shots": 1,     "max_fps": 1000,  "res_um": 0.5,   "dims": "2D",              "capital_k": 1000,  "operator": "specialist",  "solver_latency_s": 0.001},
    "coronagraphy":        {"shots": 100,   "max_fps": 0.01,  "res_um": 5,     "dims": "2D",              "capital_k": 5000,  "operator": "specialist",  "solver_latency_s": 60},
    "lucky_imaging":       {"shots": 1000,  "max_fps": 100,   "res_um": 0.3,   "dims": "2D",              "capital_k": 50,    "operator": "expert",      "solver_latency_s": 30},
    "solar_imaging":       {"shots": 1,     "max_fps": 10,    "res_um": 100,   "dims": "2D/3D",           "capital_k": 2000,  "operator": "specialist",  "solver_latency_s": 5},
    "radio_astronomy":     {"shots": 10000, "max_fps": 0.001, "res_um": 1e6,   "dims": "2D",              "capital_k": 50000, "operator": "specialist",  "solver_latency_s": 60},
    "radio_interferometry":{"shots": 10000, "max_fps": 1e-4,  "res_um": 1e4,   "dims": "2D",              "capital_k": 1e5,   "operator": "specialist",  "solver_latency_s": 3600},
    "eht_imaging":         {"shots": 10000, "max_fps": 1e-5,  "res_um": 1e4,   "dims": "2D",              "capital_k": 5e5,   "operator": "specialist",  "solver_latency_s": 86400},

    # ── X-ray Scattering / Crystallography ────────────────────────────────
    "saxs":                {"shots": 1,     "max_fps": 10,    "res_um": 1e4,   "dims": "1D/2D(recip)",    "capital_k": 500,   "operator": "expert",      "solver_latency_s": 2},
    "waxs":                {"shots": 1,     "max_fps": 10,    "res_um": 1e3,   "dims": "1D/2D(recip)",    "capital_k": 500,   "operator": "expert",      "solver_latency_s": 2},
    "xray_crystallography":{"shots": 100,   "max_fps": 0.1,   "res_um": 1e-4,  "dims": "3D(e-density)",   "capital_k": 500,   "operator": "specialist",  "solver_latency_s": 60},
    "xfel_sfx":            {"shots": 1,     "max_fps": 120,   "res_um": 1e-4,  "dims": "3D(e-density)",   "capital_k": 5e5,   "operator": "specialist",  "solver_latency_s": 3600},
    "neutron_diffraction": {"shots": 100,   "max_fps": 0.01,  "res_um": 0.001, "dims": "3D(n-density)",   "capital_k": 1e5,   "operator": "specialist",  "solver_latency_s": 300},

    # ── Neutron / Particle ────────────────────────────────────────────────
    "neutron_tomo":        {"shots": 100,   "max_fps": 0.01,  "res_um": 50,    "dims": "3D",              "capital_k": 1e5,   "operator": "specialist",  "solver_latency_s": 60},
    "particle_calorimetry":{"shots": 1,     "max_fps": 4e7,   "res_um": 1e5,   "dims": "3D(energy)",      "capital_k": 1e5,   "operator": "specialist",  "solver_latency_s": 0.001},
    "gravitational_wave":  {"shots": 1,     "max_fps": 16000, "res_um": -1,    "dims": "1D(strain)",      "capital_k": 1e6,   "operator": "specialist",  "solver_latency_s": 1},

    # ── Acoustic / Weather ────────────────────────────────────────────────
    "sonar":               {"shots": 100,   "max_fps": 1,     "res_um": 1e5,   "dims": "2D/3D",           "capital_k": 50,    "operator": "technician",  "solver_latency_s": 2},
    "ocean_acoustic_tomo": {"shots": 100,   "max_fps": 0.001, "res_um": 1e8,   "dims": "3D(sound-spd)",   "capital_k": 5000,  "operator": "specialist",  "solver_latency_s": 300},
    "weather_radar":       {"shots": 1,     "max_fps": 0.2,   "res_um": 1e6,   "dims": "3D(reflectivity)","capital_k": 1000,  "operator": "technician",  "solver_latency_s": 0.5},

    # ── Terahertz ─────────────────────────────────────────────────────────
    "terahertz":           {"shots": 100,   "max_fps": 0.1,   "res_um": 200,   "dims": "2D+spec",         "capital_k": 100,   "operator": "expert",      "solver_latency_s": 5},

    # ── Computational Photography / 3D ────────────────────────────────────
    "nerf":                {"shots": 50,    "max_fps": 0.001, "res_um": 10,    "dims": "3D(radiance)",    "capital_k": 1,     "operator": "untrained",   "solver_latency_s": 300},
    "gaussian_splatting":  {"shots": 50,    "max_fps": 30,    "res_um": 10,    "dims": "3D(radiance)",    "capital_k": 1,     "operator": "untrained",   "solver_latency_s": 120},
    "light_field":         {"shots": 1,     "max_fps": 30,    "res_um": 10,    "dims": "4D(x,y,u,v)",    "capital_k": 5,     "operator": "untrained",   "solver_latency_s": 0.5},
    "hdr_imaging":         {"shots": 3,     "max_fps": 10,    "res_um": 5,     "dims": "2D(HDR)",         "capital_k": 1,     "operator": "untrained",   "solver_latency_s": 0.1},
    "panorama":            {"shots": 10,    "max_fps": 5,     "res_um": 5,     "dims": "2D(360)",         "capital_k": 0.5,   "operator": "untrained",   "solver_latency_s": 1},
    "photometric_stereo":  {"shots": 4,     "max_fps": 5,     "res_um": 10,    "dims": "2D(normal)",      "capital_k": 2,     "operator": "technician",  "solver_latency_s": 0.5},
    "structured_light":    {"shots": 5,     "max_fps": 10,    "res_um": 50,    "dims": "3D(depth)",       "capital_k": 5,     "operator": "technician",  "solver_latency_s": 0.5},
    "tof_camera":          {"shots": 1,     "max_fps": 30,    "res_um": 1000,  "dims": "3D(depth)",       "capital_k": 2,     "operator": "untrained",   "solver_latency_s": 0.01},

    # ── Event / Ultrafast ─────────────────────────────────────────────────
    "event_camera":        {"shots": 1,     "max_fps": 1e6,   "res_um": 10,    "dims": "2D(events)",      "capital_k": 5,     "operator": "technician",  "solver_latency_s": 0.01},
    "streak_camera":       {"shots": 1,     "max_fps": 1e12,  "res_um": 50,    "dims": "1D+t",            "capital_k": 200,   "operator": "specialist",  "solver_latency_s": 0.01},
    "pump_probe":          {"shots": 1000,  "max_fps": 1e15,  "res_um": 10,    "dims": "2D+t",            "capital_k": 300,   "operator": "specialist",  "solver_latency_s": 10},

    # ── Other ─────────────────────────────────────────────────────────────
    "machine_vision":      {"shots": 1,     "max_fps": 100,   "res_um": 10,    "dims": "2D",              "capital_k": 5,     "operator": "untrained",   "solver_latency_s": 0.01},
    "clem":                {"shots": 10000, "max_fps": 1e-4,  "res_um": 0.01,  "dims": "2D(correlated)",  "capital_k": 3000,  "operator": "specialist",  "solver_latency_s": 300},
}

# ---------------------------------------------------------------------------
# Category-level defaults for modalities not in _MODALITY_HARDWARE
# ---------------------------------------------------------------------------

_CATEGORY_DEFAULTS: dict[str, dict] = {
    "compressive":          {"shots": 1,     "max_fps": 30,    "res_um": 10,    "dims": "2D/3D",       "capital_k": 20,    "operator": "expert",      "solver_latency_s": 2},
    "medical":              {"shots": 100,   "max_fps": 1,     "res_um": 500,   "dims": "3D",          "capital_k": 500,   "operator": "technician",  "solver_latency_s": 5},
    "microscopy":           {"shots": 100,   "max_fps": 10,    "res_um": 0.3,   "dims": "2D/3D",       "capital_k": 100,   "operator": "expert",      "solver_latency_s": 2},
    "coherent":             {"shots": 10,    "max_fps": 10,    "res_um": 0.5,   "dims": "2D+phase",    "capital_k": 50,    "operator": "expert",      "solver_latency_s": 1},
    "electron_microscopy":  {"shots": 1000,  "max_fps": 0.01,  "res_um": 0.001, "dims": "2D/3D",       "capital_k": 1000,  "operator": "specialist",  "solver_latency_s": 30},
    "scanning_probe":       {"shots": 1e5,   "max_fps": 0.001, "res_um": 0.01,  "dims": "2D",          "capital_k": 150,   "operator": "expert",      "solver_latency_s": 5},
    "industrial_inspection":{"shots": 1,     "max_fps": 30,    "res_um": 100,   "dims": "2D",          "capital_k": 30,    "operator": "technician",  "solver_latency_s": 1},
    "experimental_science": {"shots": 100,   "max_fps": 1,     "res_um": 10,    "dims": "2D",          "capital_k": 100,   "operator": "expert",      "solver_latency_s": 5},
    "remote_sensing":       {"shots": 1,     "max_fps": 0.1,   "res_um": 1e6,   "dims": "2D",          "capital_k": 1000,  "operator": "specialist",  "solver_latency_s": 10},
    "computational":        {"shots": 10,    "max_fps": 10,    "res_um": 10,    "dims": "2D/3D",       "capital_k": 5,     "operator": "technician",  "solver_latency_s": 1},
    "depth_imaging":        {"shots": 1,     "max_fps": 30,    "res_um": 100,   "dims": "3D(depth)",   "capital_k": 5,     "operator": "technician",  "solver_latency_s": 0.5},
    "scientific_instrumentation": {"shots": 100, "max_fps": 0.1, "res_um": 1, "dims": "2D/3D", "capital_k": 500, "operator": "specialist", "solver_latency_s": 30},
    "particle_imaging":     {"shots": 1e7,   "max_fps": 0.001, "res_um": 3000,  "dims": "3D",          "capital_k": 1000,  "operator": "technician",  "solver_latency_s": 30},
    "astronomy":            {"shots": 100,   "max_fps": 1,     "res_um": 1,     "dims": "2D",          "capital_k": 1000,  "operator": "specialist",  "solver_latency_s": 30},
    "multi_modal_fusion":   {"shots": 100,   "max_fps": 1,     "res_um": 500,   "dims": "3D(fused)",   "capital_k": 200,   "operator": "expert",      "solver_latency_s": 5},
    "clinical_optics":      {"shots": 1,     "max_fps": 30,    "res_um": 10,    "dims": "2D/3D",       "capital_k": 50,    "operator": "technician",  "solver_latency_s": 1},
    "medical_ultrasound":   {"shots": 1,     "max_fps": 50,    "res_um": 300,   "dims": "2D",          "capital_k": 50,    "operator": "technician",  "solver_latency_s": 0.01},
    "fiber_endoscopy":      {"shots": 1,     "max_fps": 30,    "res_um": 10,    "dims": "2D",          "capital_k": 50,    "operator": "technician",  "solver_latency_s": 0.1},
}

# Sample compatibility: carrier → (contact, destructive, in_vivo)
_CARRIER_SAMPLE_PROPS: dict[str, dict] = {
    "Photon":       {"contact": False, "destructive": False, "in_vivo": True},
    "X-ray":        {"contact": False, "destructive": False, "in_vivo": True},
    "Spin/RF":      {"contact": False, "destructive": False, "in_vivo": True},
    "Acoustic":     {"contact": True,  "destructive": False, "in_vivo": True},
    "IR":           {"contact": False, "destructive": False, "in_vivo": False},
    "Electron":     {"contact": False, "destructive": True,  "in_vivo": False},
    "Gamma":        {"contact": False, "destructive": False, "in_vivo": True},
    "Mechanical":   {"contact": True,  "destructive": False, "in_vivo": False},
    "Ion":          {"contact": False, "destructive": True,  "in_vivo": False},
    "MV":           {"contact": False, "destructive": False, "in_vivo": True},
    "Proton":       {"contact": False, "destructive": False, "in_vivo": True},
    "Gamma/X-ray":  {"contact": False, "destructive": False, "in_vivo": True},
    "Neutron":      {"contact": False, "destructive": False, "in_vivo": False},
    "RF":           {"contact": False, "destructive": False, "in_vivo": False},
    "Microwave":    {"contact": False, "destructive": False, "in_vivo": False},
    "THz":          {"contact": False, "destructive": False, "in_vivo": False},
    "Gravitational":{"contact": False, "destructive": False, "in_vivo": False},
    "Muon":         {"contact": False, "destructive": False, "in_vivo": False},
    "Seismic":      {"contact": False, "destructive": False, "in_vivo": False},
}


def _get_hardware(modality_id: str, category: str) -> dict:
    """Get hardware properties for a modality, falling back to category defaults."""
    if modality_id in _MODALITY_HARDWARE:
        return dict(_MODALITY_HARDWARE[modality_id])
    if category in _CATEGORY_DEFAULTS:
        return dict(_CATEGORY_DEFAULTS[category])
    return dict(_CATEGORY_DEFAULTS.get("experimental_science", {}))


def _get_best_scores(modality_id: str, category: str) -> dict:
    """Get best PSNR/SSIM for a modality from CATEGORY_REAL_SCORES."""
    score_key = get_score_key(modality_id, category)
    scores = CATEGORY_REAL_SCORES.get(score_key, [])
    if not scores:
        return {"best_psnr": None, "best_ssim": None, "best_method": None,
                "worst_psnr": None, "num_methods": 0}
    best = max(scores, key=lambda s: s.get("psnr", 0))
    worst = min(scores, key=lambda s: s.get("psnr", 999))
    return {
        "best_psnr": best.get("psnr"),
        "best_ssim": best.get("ssim"),
        "best_method": best.get("method"),
        "best_source": best.get("source"),
        "worst_psnr": worst.get("psnr"),
        "num_methods": len(scores),
        "score_key": score_key,
    }


def _get_solver_info(modality_id: str, category: str) -> dict:
    """Get the best algorithm info for a modality."""
    algos = get_algorithms(modality_id, category)
    if not algos:
        return {"solver_name": "Unknown", "solver_type": "Unknown", "solver_source": ""}

    # Prefer the highest-quality type in this order
    type_priority = [
        "Diffusion", "Score-based", "Vision Transformer", "Transformer",
        "Deep Unrolling", "Deep Learning", "Physics-Informed",
        "PnP", "Compressed Sensing", "Classical",
    ]
    best = algos[-1]  # last is usually best
    for t in type_priority:
        for a in algos:
            if a.get("type") == t:
                best = a
                break
        if best.get("type") == t:
            break

    return {
        "solver_name": best.get("name", "Unknown"),
        "solver_type": best.get("type", "Unknown"),
        "solver_source": best.get("source", ""),
        "solver_params": best.get("params", "0"),
        "num_algorithms": len(algos),
        "algorithm_types": sorted(set(a.get("type", "") for a in algos)),
    }


def _sample_props(carrier: str, modality_id: str) -> dict:
    """Get sample compatibility properties."""
    props = _CARRIER_SAMPLE_PROPS.get(carrier, {"contact": False, "destructive": False, "in_vivo": False})
    # Per-modality overrides
    if modality_id in ("atom_probe", "fib_sem"):
        props = {"contact": False, "destructive": True, "in_vivo": False}
    elif modality_id in ("maldi_msi", "sims"):
        props = {"contact": False, "destructive": True, "in_vivo": False}
    elif modality_id in ("afm", "stm", "nsom", "mfm"):
        props = {"contact": True, "destructive": False, "in_vivo": False}
    elif modality_id in ("ultrasound", "doppler_ultrasound", "elastography", "ceus", "ivus"):
        props = {"contact": True, "destructive": False, "in_vivo": True}
    return props


def generate_system_descriptor(modality_id: str, mod_entry: dict) -> dict:
    """Generate a complete Layer A system descriptor for one modality."""
    category = mod_entry["category"]
    carrier = mod_entry.get("carrier", "Unknown")
    display_name = mod_entry["display_name"]

    hw = _get_hardware(modality_id, category)
    scores = _get_best_scores(modality_id, category)
    solver = _get_solver_info(modality_id, category)
    sample = _sample_props(carrier, modality_id)

    mismatch_params = mod_entry.get("mismatch_params", [])

    return {
        # Identity
        "id": modality_id,
        "display_name": display_name,
        "category": category,
        "carrier": carrier,

        # Physical chain
        "spec_notation": mod_entry.get("spec_notation", ""),
        "canonical_dag": mod_entry.get("canonical_dag", ""),
        "primitives": mod_entry.get("primitives", []),
        "has_dedicated_operator": mod_entry.get("has_dedicated_operator", False),

        # Mismatch parameters (from YAML configs)
        "num_mismatch_params": len(mismatch_params),
        "mismatch_params": [
            {"name": p["name"], "symbol": p["symbol"], "description": p["description"]}
            for p in mismatch_params
        ],

        # Hardware properties
        "shots_per_datacube": hw.get("shots", 1),
        "max_fps": hw.get("max_fps", 1),
        "spatial_resolution_um": hw.get("res_um", 10),
        "output_dimensionality": hw.get("dims", "2D"),
        "capital_cost_k_usd": hw.get("capital_k", 100),
        "operator_skill": hw.get("operator", "expert"),

        # Solver properties (auto-populated from algorithm catalog)
        "solver_name": solver["solver_name"],
        "solver_type": solver["solver_type"],
        "solver_source": solver["solver_source"],
        "solver_params": solver["solver_params"],
        "solver_latency_s": hw.get("solver_latency_s", 5),
        "num_algorithms_in_catalog": solver["num_algorithms"],
        "algorithm_type_coverage": solver["algorithm_types"],

        # Benchmark scores (auto-populated from CATEGORY_REAL_SCORES)
        "best_psnr_db": scores.get("best_psnr"),
        "best_ssim": scores.get("best_ssim"),
        "best_method": scores.get("best_method"),
        "best_method_source": scores.get("best_source"),
        "worst_psnr_db": scores.get("worst_psnr"),
        "num_benchmark_results": scores.get("num_methods", 0),
        "score_routing_key": scores.get("score_key"),

        # Sample compatibility
        "sample_contact": sample["contact"],
        "sample_destructive": sample["destructive"],
        "in_vivo_capable": sample["in_vivo"],

        # Confidence levels
        "confidence": {
            "hardware": "verified" if modality_id in _MODALITY_HARDWARE else "estimated",
            "algorithms": "verified",  # auto-populated from existing catalog
            "scores": "verified" if scores.get("num_methods", 0) > 0 else "unknown",
        },
    }


def main():
    print(f"Generating system catalog for {len(MODALITY_CATALOG)} modalities...")

    catalog = {}
    stats = {"verified_hw": 0, "estimated_hw": 0, "with_scores": 0, "total_algos": 0}

    for mod_id, mod_entry in sorted(MODALITY_CATALOG.items()):
        descriptor = generate_system_descriptor(mod_id, mod_entry)
        catalog[mod_id] = descriptor

        if descriptor["confidence"]["hardware"] == "verified":
            stats["verified_hw"] += 1
        else:
            stats["estimated_hw"] += 1
        if descriptor["num_benchmark_results"] > 0:
            stats["with_scores"] += 1
        stats["total_algos"] += descriptor["num_algorithms_in_catalog"]

    # Write output
    out_dir = _PLATFORM_DIR / "pwm_platform" / "static" / "benchmark-data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "system_catalog.json"
    with open(out_path, "w") as f:
        json.dump(catalog, f, indent=2, default=str)

    print(f"\nSystem catalog written to: {out_path}")
    print(f"  Total modalities:       {len(catalog)}")
    print(f"  Verified hardware:      {stats['verified_hw']}")
    print(f"  Estimated hardware:     {stats['estimated_hw']}")
    print(f"  With benchmark scores:  {stats['with_scores']}")
    print(f"  Total algorithm entries: {stats['total_algos']}")

    # Summary by category
    cat_counts: dict[str, int] = {}
    for d in catalog.values():
        cat_counts[d["category"]] = cat_counts.get(d["category"], 0) + 1
    print("\n  By category:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:30s} {count:3d}")

    # List modalities missing hardware data
    missing = [d["id"] for d in catalog.values()
               if d["confidence"]["hardware"] == "estimated"]
    if missing:
        print(f"\n  Modalities using category defaults ({len(missing)}):")
        for m in missing:
            print(f"    - {m}")

    return catalog


if __name__ == "__main__":
    main()
