"""Compact metadata for all 65 benchmark variants.

Each entry contains only variant-specific data (spec DAG, mismatch params).
The factory function expands these into full VARIANT_DATABASE entries at
import time.
"""

from __future__ import annotations

# fmt: off

VARIANT_REGISTRY: dict[str, dict] = {

    # ══════════════════════════════════════════════════════════════════════════
    #  COMPRESSIVE  (5 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "sd_cassi": {
        "display_name": "CASSI",
        "full_name": "Coded Aperture Snapshot Spectral Imaging",
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
            {"name": "sigma_read", "symbol": "\u03c3_r", "description": "Detector read noise std (electrons)", "nominal": 5.0, "perturbed": 8.0},
            {"name": "dark_current", "symbol": "I_d", "description": "Dark current (electrons/pixel/s)", "nominal": 0.1, "perturbed": 0.5},
            {"name": "gain", "symbol": "g", "description": "Detector gain multiplier", "nominal": 1.0, "perturbed": 1.03},
        ],
    },

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
    },

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
    },

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
    },

    "matrix": {
        "display_name": "Matrix",
        "full_name": "Generic Matrix Sensing",
        "parent_modality": "matrix",
        "category": "compressive",
        "spec_notation": "M(\u03a6) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "M", "params": "\u03a6", "label": "Sensing Matrix"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Detector"},
        ],
        "mismatch_params": [
            {"name": "matrix_perturb", "symbol": "\u0394\u03a6", "description": "Matrix element perturbation std", "nominal": 0, "perturbed": 0.01},
            {"name": "gain", "symbol": "g", "description": "Detector gain multiplier", "nominal": 1.0, "perturbed": 1.03},
            {"name": "sigma_y", "symbol": "\u03c3_y", "description": "Measurement noise std", "nominal": 0, "perturbed": 0.02},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  MEDICAL — Projection-based  (11 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "ct": {
        "display_name": "CT",
        "full_name": "X-ray Computed Tomography",
        "parent_modality": "ct",
        "category": "medical",
        "spec_notation": "R(\u03b8) \u2192 \u03a0(fan) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "R", "params": "\u03b8", "label": "Gantry Rotation"},
            {"primitive": "Pi", "params": "fan", "label": "Fan-Beam Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Detector"},
        ],
        "mismatch_params": [
            {"name": "center_offset", "symbol": "\u0394c", "description": "Center-of-rotation offset (pixels)", "nominal": 0, "perturbed": 1.5},
            {"name": "angle_error", "symbol": "\u0394\u03b8", "description": "Gantry angle error (deg)", "nominal": 0, "perturbed": 0.5},
            {"name": "beam_hardening", "symbol": "\u03b2", "description": "Beam-hardening coefficient", "nominal": 0, "perturbed": 0.03},
            {"name": "detector_tilt", "symbol": "\u03c6", "description": "Detector tilt (deg)", "nominal": 0, "perturbed": 0.2},
        ],
    },

    "cbct": {
        "display_name": "CBCT",
        "full_name": "Cone-Beam Computed Tomography",
        "parent_modality": "cbct",
        "category": "medical",
        "spec_notation": "R(\u03b8) \u2192 \u03a0(cone) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "R", "params": "\u03b8", "label": "Gantry Rotation"},
            {"primitive": "Pi", "params": "cone", "label": "Cone-Beam Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Detector"},
        ],
        "mismatch_params": [
            {"name": "center_offset", "symbol": "\u0394c", "description": "Center-of-rotation offset (pixels)", "nominal": 0, "perturbed": 2.0},
            {"name": "source_dist", "symbol": "\u0394d_s", "description": "Source-to-isocenter distance error (mm)", "nominal": 0, "perturbed": 1.0},
            {"name": "cone_angle", "symbol": "\u0394\u03b1", "description": "Cone half-angle error (deg)", "nominal": 0, "perturbed": 0.3},
            {"name": "detector_tilt", "symbol": "\u03c6", "description": "Detector tilt (deg)", "nominal": 0, "perturbed": 0.5},
        ],
    },

    "pet": {
        "display_name": "PET",
        "full_name": "Positron Emission Tomography",
        "parent_modality": "pet",
        "category": "medical",
        "spec_notation": "\u03a0(LOR) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "Pi", "params": "LOR", "label": "Line-of-Response Projection"},
            {"primitive": "Sigma", "params": "t", "label": "Temporal Integration"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "Scintillation Detector"},
        ],
        "mismatch_params": [
            {"name": "attenuation", "symbol": "\u03bc", "description": "Attenuation map error (%)", "nominal": 0, "perturbed": 5.0},
            {"name": "scatter_frac", "symbol": "f_s", "description": "Scatter fraction error", "nominal": 0.3, "perturbed": 0.35},
            {"name": "timing_res", "symbol": "\u0394t", "description": "Timing resolution jitter (ps)", "nominal": 200, "perturbed": 250},
            {"name": "normalization", "symbol": "n", "description": "Detector normalization error (%)", "nominal": 0, "perturbed": 2.0},
        ],
    },

    "spect": {
        "display_name": "SPECT",
        "full_name": "Single Photon Emission Computed Tomography",
        "parent_modality": "spect",
        "category": "medical",
        "spec_notation": "\u03a0(parallel) \u2192 \u03a3_E \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "Pi", "params": "parallel", "label": "Parallel-Hole Collimator"},
            {"primitive": "Sigma", "params": "E", "label": "Energy Window Sum"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "Gamma Camera"},
        ],
        "mismatch_params": [
            {"name": "center_offset", "symbol": "\u0394c", "description": "Center-of-rotation offset (pixels)", "nominal": 0, "perturbed": 1.5},
            {"name": "collimator_septal", "symbol": "s", "description": "Septal penetration fraction", "nominal": 0, "perturbed": 0.02},
            {"name": "attenuation", "symbol": "\u03bc", "description": "Attenuation coefficient error (%)", "nominal": 0, "perturbed": 5.0},
            {"name": "scatter", "symbol": "f_s", "description": "Scatter fraction error", "nominal": 0.2, "perturbed": 0.25},
        ],
    },

    "xray_radiography": {
        "display_name": "X-ray Radiography",
        "full_name": "X-ray Radiography",
        "parent_modality": "xray_radiography",
        "category": "medical",
        "spec_notation": "\u03a0(proj) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "Pi", "params": "proj", "label": "X-ray Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Flat-Panel Detector"},
        ],
        "mismatch_params": [
            {"name": "source_dist", "symbol": "\u0394d", "description": "Source-to-detector distance error (mm)", "nominal": 0, "perturbed": 5.0},
            {"name": "beam_hardening", "symbol": "\u03b2", "description": "Beam-hardening coefficient", "nominal": 0, "perturbed": 0.02},
            {"name": "scatter", "symbol": "f_s", "description": "Scatter-to-primary ratio error", "nominal": 0, "perturbed": 0.05},
        ],
    },

    "mammography": {
        "display_name": "Mammography",
        "full_name": "Mammography",
        "parent_modality": "mammography",
        "category": "medical",
        "spec_notation": "\u03a0(contact) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "Pi", "params": "contact", "label": "Contact Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Detector"},
        ],
        "mismatch_params": [
            {"name": "compression", "symbol": "\u0394h", "description": "Compression thickness error (mm)", "nominal": 0, "perturbed": 2.0},
            {"name": "anode_angle", "symbol": "\u0394\u03b1", "description": "Anode angle error (deg)", "nominal": 0, "perturbed": 0.5},
            {"name": "scatter", "symbol": "f_s", "description": "Scatter fraction error", "nominal": 0.3, "perturbed": 0.35},
        ],
    },

    "fluoroscopy": {
        "display_name": "Fluoroscopy",
        "full_name": "Fluoroscopy",
        "parent_modality": "fluoroscopy",
        "category": "medical",
        "spec_notation": "\u03a0(proj) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "Pi", "params": "proj", "label": "X-ray Projection"},
            {"primitive": "Sigma", "params": "t", "label": "Temporal Integration"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Image Intensifier"},
        ],
        "mismatch_params": [
            {"name": "motion_blur", "symbol": "\u03c3_t", "description": "Temporal motion blur (ms)", "nominal": 0, "perturbed": 5.0},
            {"name": "lag", "symbol": "\u03c4", "description": "Detector lag time constant (ms)", "nominal": 0, "perturbed": 3.0},
            {"name": "gain_drift", "symbol": "\u0394g", "description": "Gain drift per frame (%)", "nominal": 0, "perturbed": 0.5},
        ],
    },

    "angiography": {
        "display_name": "X-ray Angiography",
        "full_name": "X-ray Angiography",
        "parent_modality": "angiography",
        "category": "medical",
        "spec_notation": "\u03a0(proj) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "Pi", "params": "proj", "label": "X-ray Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Flat-Panel Detector"},
        ],
        "mismatch_params": [
            {"name": "contrast_timing", "symbol": "\u0394t_c", "description": "Contrast bolus timing error (s)", "nominal": 0, "perturbed": 0.5},
            {"name": "motion", "symbol": "\u03c3_m", "description": "Patient motion (mm)", "nominal": 0, "perturbed": 2.0},
            {"name": "scatter", "symbol": "f_s", "description": "Scatter fraction error", "nominal": 0, "perturbed": 0.05},
        ],
    },

    "dexa": {
        "display_name": "DEXA",
        "full_name": "Dual-Energy X-ray Absorptiometry",
        "parent_modality": "dexa",
        "category": "medical",
        "spec_notation": "\u039b(E\u2081,E\u2082) \u2192 \u03a0(proj) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "Lambda", "params": "E\u2081,E\u2082", "label": "Dual-Energy Selection"},
            {"primitive": "Pi", "params": "proj", "label": "X-ray Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Detector"},
        ],
        "mismatch_params": [
            {"name": "energy_offset", "symbol": "\u0394E", "description": "Energy calibration offset (keV)", "nominal": 0, "perturbed": 1.0},
            {"name": "soft_tissue", "symbol": "\u0394\u03bc_s", "description": "Soft-tissue attenuation error (%)", "nominal": 0, "perturbed": 3.0},
            {"name": "beam_overlap", "symbol": "f_o", "description": "Spectral overlap fraction", "nominal": 0, "perturbed": 0.02},
        ],
    },

    "dot": {
        "display_name": "DOT",
        "full_name": "Diffuse Optical Tomography",
        "parent_modality": "dot",
        "category": "medical",
        "spec_notation": "P(diffuse) \u2192 \u03a3 \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "P", "params": "diffuse", "label": "Diffuse Propagation"},
            {"primitive": "Sigma", "params": "", "label": "Boundary Integration"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "Photodetector"},
        ],
        "mismatch_params": [
            {"name": "mu_a", "symbol": "\u0394\u03bc_a", "description": "Absorption coefficient error (%)", "nominal": 0, "perturbed": 10.0},
            {"name": "mu_s", "symbol": "\u0394\u03bc_s'", "description": "Reduced scattering error (%)", "nominal": 0, "perturbed": 8.0},
            {"name": "source_pos", "symbol": "\u0394r_s", "description": "Source position error (mm)", "nominal": 0, "perturbed": 1.0},
        ],
    },

    "photoacoustic": {
        "display_name": "Photoacoustic",
        "full_name": "Photoacoustic Imaging",
        "parent_modality": "photoacoustic",
        "category": "medical",
        "spec_notation": "P(acoustic) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2082)",
        "spec_dag": [
            {"primitive": "P", "params": "acoustic", "label": "Acoustic Propagation"},
            {"primitive": "Sigma", "params": "t", "label": "Temporal Integration"},
            {"primitive": "D", "params": "g, \u03b7\u2082", "label": "Ultrasound Transducer"},
        ],
        "mismatch_params": [
            {"name": "sos", "symbol": "\u0394c", "description": "Speed-of-sound error (m/s)", "nominal": 1540, "perturbed": 1560},
            {"name": "fluence", "symbol": "\u0394\u03a6", "description": "Fluence distribution error (%)", "nominal": 0, "perturbed": 10.0},
            {"name": "sensor_response", "symbol": "\u0394h", "description": "Sensor impulse response error (%)", "nominal": 0, "perturbed": 5.0},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  MEDICAL — Fourier-based  (4 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "mri": {
        "display_name": "MRI",
        "full_name": "Magnetic Resonance Imaging",
        "parent_modality": "mri",
        "category": "medical",
        "spec_notation": "F(k-traj) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "F", "params": "k-traj", "label": "k-Space Sampling"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "RF Coil Receiver"},
        ],
        "mismatch_params": [
            {"name": "B0_inhomog", "symbol": "\u0394B\u2080", "description": "B\u2080 field inhomogeneity (ppm)", "nominal": 0, "perturbed": 1.5},
            {"name": "gradient_nonlin", "symbol": "\u0394G", "description": "Gradient nonlinearity (%)", "nominal": 0, "perturbed": 2.0},
            {"name": "coil_sensitivity", "symbol": "\u0394S", "description": "Coil sensitivity map error (%)", "nominal": 0, "perturbed": 5.0},
            {"name": "k_trajectory", "symbol": "\u0394k", "description": "k-space trajectory error (%)", "nominal": 0, "perturbed": 1.0},
        ],
    },

    "fmri": {
        "display_name": "fMRI",
        "full_name": "Functional MRI (BOLD)",
        "parent_modality": "fmri",
        "category": "medical",
        "spec_notation": "F(EPI) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "F", "params": "EPI", "label": "EPI k-Space Readout"},
            {"primitive": "Sigma", "params": "t", "label": "Temporal Averaging"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "RF Coil Receiver"},
        ],
        "mismatch_params": [
            {"name": "B0_inhomog", "symbol": "\u0394B\u2080", "description": "B\u2080 field inhomogeneity (ppm)", "nominal": 0, "perturbed": 2.0},
            {"name": "head_motion", "symbol": "\u0394r", "description": "Head motion (mm)", "nominal": 0, "perturbed": 1.0},
            {"name": "hemodynamic_delay", "symbol": "\u0394\u03c4", "description": "HRF delay error (s)", "nominal": 6.0, "perturbed": 7.0},
            {"name": "physiological_noise", "symbol": "\u03c3_p", "description": "Physiological noise amplitude", "nominal": 0, "perturbed": 0.02},
        ],
    },

    "diffusion_mri": {
        "display_name": "Diffusion MRI",
        "full_name": "Diffusion MRI (DTI)",
        "parent_modality": "diffusion_mri",
        "category": "medical",
        "spec_notation": "F(EPI) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "F", "params": "EPI", "label": "Diffusion-Weighted EPI"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "RF Coil Receiver"},
        ],
        "mismatch_params": [
            {"name": "b_value_error", "symbol": "\u0394b", "description": "b-value calibration error (%)", "nominal": 0, "perturbed": 3.0},
            {"name": "eddy_current", "symbol": "\u03b5", "description": "Eddy-current distortion (voxels)", "nominal": 0, "perturbed": 0.5},
            {"name": "gradient_direction", "symbol": "\u0394g\u0302", "description": "Gradient direction error (deg)", "nominal": 0, "perturbed": 1.0},
            {"name": "susceptibility", "symbol": "\u0394\u03c7", "description": "Susceptibility distortion (voxels)", "nominal": 0, "perturbed": 1.0},
        ],
    },

    "mrs": {
        "display_name": "MRS",
        "full_name": "MR Spectroscopy",
        "parent_modality": "mrs",
        "category": "medical",
        "spec_notation": "F(FID) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "F", "params": "FID", "label": "Free Induction Decay"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "RF Coil Receiver"},
        ],
        "mismatch_params": [
            {"name": "linewidth", "symbol": "\u0394\u03bd", "description": "Linewidth broadening (Hz)", "nominal": 0, "perturbed": 2.0},
            {"name": "freq_drift", "symbol": "\u0394f", "description": "Frequency drift (Hz)", "nominal": 0, "perturbed": 1.5},
            {"name": "phase_error", "symbol": "\u0394\u03c6", "description": "Zero-order phase error (deg)", "nominal": 0, "perturbed": 5.0},
            {"name": "baseline", "symbol": "B", "description": "Baseline distortion amplitude", "nominal": 0, "perturbed": 0.05},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  MEDICAL ULTRASOUND  (3 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "ultrasound": {
        "display_name": "Ultrasound",
        "full_name": "Ultrasound Imaging",
        "parent_modality": "ultrasound",
        "category": "medical_ultrasound",
        "spec_notation": "P(acoustic) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2082)",
        "spec_dag": [
            {"primitive": "P", "params": "acoustic", "label": "Acoustic Propagation"},
            {"primitive": "Sigma", "params": "t", "label": "Temporal Integration"},
            {"primitive": "D", "params": "g, \u03b7\u2082", "label": "Piezo Array"},
        ],
        "mismatch_params": [
            {"name": "sos", "symbol": "\u0394c", "description": "Speed-of-sound error (m/s)", "nominal": 1540, "perturbed": 1560},
            {"name": "attenuation", "symbol": "\u0394\u03b1", "description": "Attenuation coefficient error (dB/cm/MHz)", "nominal": 0.5, "perturbed": 0.6},
            {"name": "element_sensitivity", "symbol": "\u0394s", "description": "Element sensitivity variation (%)", "nominal": 0, "perturbed": 5.0},
            {"name": "phase_aberration", "symbol": "\u0394\u03c6", "description": "Phase aberration (rad)", "nominal": 0, "perturbed": 0.3},
        ],
    },

    "doppler_ultrasound": {
        "display_name": "Doppler Ultrasound",
        "full_name": "Doppler Ultrasound",
        "parent_modality": "doppler_ultrasound",
        "category": "medical_ultrasound",
        "spec_notation": "P(acoustic) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2082)",
        "spec_dag": [
            {"primitive": "P", "params": "acoustic", "label": "Acoustic Propagation"},
            {"primitive": "Sigma", "params": "t", "label": "Temporal Integration"},
            {"primitive": "D", "params": "g, \u03b7\u2082", "label": "Piezo Array"},
        ],
        "mismatch_params": [
            {"name": "sos", "symbol": "\u0394c", "description": "Speed-of-sound error (m/s)", "nominal": 1540, "perturbed": 1555},
            {"name": "doppler_angle", "symbol": "\u0394\u03b8", "description": "Doppler angle error (deg)", "nominal": 0, "perturbed": 5.0},
            {"name": "wall_filter", "symbol": "\u0394f_w", "description": "Wall filter cutoff error (Hz)", "nominal": 50, "perturbed": 80},
            {"name": "prf", "symbol": "\u0394PRF", "description": "PRF jitter (%)", "nominal": 0, "perturbed": 1.0},
        ],
    },

    "elastography": {
        "display_name": "Elastography",
        "full_name": "Shear-Wave Elastography",
        "parent_modality": "elastography",
        "category": "medical_ultrasound",
        "spec_notation": "P(shear) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2082)",
        "spec_dag": [
            {"primitive": "P", "params": "shear", "label": "Shear-Wave Propagation"},
            {"primitive": "Sigma", "params": "t", "label": "Temporal Tracking"},
            {"primitive": "D", "params": "g, \u03b7\u2082", "label": "Ultrafast Array"},
        ],
        "mismatch_params": [
            {"name": "shear_speed", "symbol": "\u0394c_s", "description": "Shear speed error (m/s)", "nominal": 0, "perturbed": 0.3},
            {"name": "push_duration", "symbol": "\u0394\u03c4", "description": "Push-pulse duration error (\u03bcs)", "nominal": 0, "perturbed": 10},
            {"name": "tissue_viscosity", "symbol": "\u0394\u03b7", "description": "Viscosity model error (%)", "nominal": 0, "perturbed": 15.0},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  COHERENT  (3 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "holography": {
        "display_name": "Holography",
        "full_name": "Digital Holographic Microscopy",
        "parent_modality": "holography",
        "category": "coherent",
        "spec_notation": "P(Fresnel) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "Fresnel", "label": "Fresnel Propagation"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Camera"},
        ],
        "mismatch_params": [
            {"name": "wavelength", "symbol": "\u0394\u03bb", "description": "Wavelength error (nm)", "nominal": 0, "perturbed": 0.5},
            {"name": "prop_distance", "symbol": "\u0394z", "description": "Propagation distance error (\u03bcm)", "nominal": 0, "perturbed": 5.0},
            {"name": "tilt", "symbol": "\u0394\u03b8", "description": "Reference beam tilt (mrad)", "nominal": 0, "perturbed": 0.5},
        ],
    },

    "ptychography": {
        "display_name": "Ptychography",
        "full_name": "Ptychographic Imaging",
        "parent_modality": "ptychography",
        "category": "coherent",
        "spec_notation": "P(probe) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "probe", "label": "Probe Illumination + Propagation"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Diffraction Detector"},
        ],
        "mismatch_params": [
            {"name": "probe_error", "symbol": "\u0394P", "description": "Probe function error (%)", "nominal": 0, "perturbed": 5.0},
            {"name": "position_error", "symbol": "\u0394r", "description": "Scan position error (nm)", "nominal": 0, "perturbed": 10.0},
            {"name": "partial_coherence", "symbol": "\u03c3_c", "description": "Partial coherence width (nm)", "nominal": 0, "perturbed": 5.0},
        ],
    },

    "phase_retrieval": {
        "display_name": "CDI",
        "full_name": "Coherent Diffractive Imaging",
        "parent_modality": "phase_retrieval",
        "category": "coherent",
        "spec_notation": "P(far-field) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "far-field", "label": "Far-Field Propagation"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Diffraction Detector"},
        ],
        "mismatch_params": [
            {"name": "support", "symbol": "\u0394S", "description": "Support constraint error (pixels)", "nominal": 0, "perturbed": 3.0},
            {"name": "saturation", "symbol": "I_sat", "description": "Detector saturation threshold error (%)", "nominal": 0, "perturbed": 5.0},
            {"name": "missing_center", "symbol": "r_0", "description": "Missing center radius (pixels)", "nominal": 0, "perturbed": 3},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  MICROSCOPY — PSF-based  (9 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "widefield": {
        "display_name": "Widefield",
        "full_name": "Widefield Fluorescence Microscopy",
        "parent_modality": "widefield",
        "category": "microscopy",
        "spec_notation": "C(PSF) \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF", "label": "PSF Convolution"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "sCMOS Camera"},
        ],
        "mismatch_params": [
            {"name": "psf_sigma", "symbol": "\u0394\u03c3", "description": "PSF width error (%)", "nominal": 0, "perturbed": 10.0},
            {"name": "defocus", "symbol": "\u0394z", "description": "Defocus error (\u03bcm)", "nominal": 0, "perturbed": 0.5},
            {"name": "background", "symbol": "\u0394b", "description": "Background fluorescence offset", "nominal": 0, "perturbed": 50},
        ],
    },

    "widefield_lowdose": {
        "display_name": "Widefield Low-Dose",
        "full_name": "Low-Dose Widefield Microscopy",
        "parent_modality": "widefield_lowdose",
        "category": "microscopy",
        "spec_notation": "C(PSF) \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF", "label": "PSF Convolution"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "sCMOS Camera"},
        ],
        "mismatch_params": [
            {"name": "psf_sigma", "symbol": "\u0394\u03c3", "description": "PSF width error (%)", "nominal": 0, "perturbed": 10.0},
            {"name": "photon_budget", "symbol": "\u0394N", "description": "Photon budget error (%)", "nominal": 0, "perturbed": 20.0},
            {"name": "read_noise", "symbol": "\u0394\u03c3_r", "description": "Read noise error (e-)", "nominal": 1.5, "perturbed": 2.5},
        ],
    },

    "confocal_livecell": {
        "display_name": "Confocal Live-Cell",
        "full_name": "Confocal Live-Cell Microscopy",
        "parent_modality": "confocal_livecell",
        "category": "microscopy",
        "spec_notation": "C(PSF_confocal) \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF_confocal", "label": "Confocal PSF"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "PMT / HyD"},
        ],
        "mismatch_params": [
            {"name": "pinhole", "symbol": "\u0394ph", "description": "Pinhole diameter error (\u03bcm)", "nominal": 0, "perturbed": 5.0},
            {"name": "refractive_index", "symbol": "\u0394n", "description": "Refractive index mismatch", "nominal": 1.515, "perturbed": 1.52},
            {"name": "photobleaching", "symbol": "\u03b1_b", "description": "Photobleaching rate error (%)", "nominal": 0, "perturbed": 5.0},
        ],
    },

    "confocal_3d": {
        "display_name": "Confocal 3D",
        "full_name": "Confocal 3D Z-Stack",
        "parent_modality": "confocal_3d",
        "category": "microscopy",
        "spec_notation": "C(PSF_3D) \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF_3D", "label": "3D Confocal PSF"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "PMT / HyD"},
        ],
        "mismatch_params": [
            {"name": "z_step", "symbol": "\u0394z", "description": "Z-step size error (nm)", "nominal": 0, "perturbed": 50},
            {"name": "spherical_aberr", "symbol": "C_s", "description": "Spherical aberration (waves)", "nominal": 0, "perturbed": 0.1},
            {"name": "refractive_index", "symbol": "\u0394n", "description": "Refractive index mismatch", "nominal": 1.515, "perturbed": 1.525},
        ],
    },

    "lightsheet": {
        "display_name": "Light-Sheet",
        "full_name": "Light-Sheet Fluorescence Microscopy",
        "parent_modality": "lightsheet",
        "category": "microscopy",
        "spec_notation": "C(PSF_sheet) \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF_sheet", "label": "Light-Sheet PSF"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "sCMOS Camera"},
        ],
        "mismatch_params": [
            {"name": "sheet_thickness", "symbol": "\u0394w", "description": "Sheet thickness error (\u03bcm)", "nominal": 0, "perturbed": 1.0},
            {"name": "sheet_tilt", "symbol": "\u0394\u03b8", "description": "Sheet tilt (deg)", "nominal": 0, "perturbed": 0.5},
            {"name": "stripe_artifact", "symbol": "\u03b1_s", "description": "Stripe artifact amplitude", "nominal": 0, "perturbed": 0.1},
        ],
    },

    "two_photon": {
        "display_name": "Two-Photon",
        "full_name": "Two-Photon / Multiphoton Microscopy",
        "parent_modality": "two_photon",
        "category": "microscopy",
        "spec_notation": "C(PSF_2P) \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF_2P", "label": "Two-Photon PSF"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "Non-Descanned PMT"},
        ],
        "mismatch_params": [
            {"name": "pulse_width", "symbol": "\u0394\u03c4", "description": "Pulse width broadening (fs)", "nominal": 100, "perturbed": 140},
            {"name": "gdd", "symbol": "\u0394GDD", "description": "Group delay dispersion error (fs\u00b2)", "nominal": 0, "perturbed": 500},
            {"name": "scattering", "symbol": "\u0394\u03bc_s", "description": "Tissue scattering error (%)", "nominal": 0, "perturbed": 10.0},
        ],
    },

    "sted": {
        "display_name": "STED",
        "full_name": "STED Microscopy",
        "parent_modality": "sted",
        "category": "microscopy",
        "spec_notation": "C(PSF_STED) \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF_STED", "label": "STED Effective PSF"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "Avalanche Photodiode"},
        ],
        "mismatch_params": [
            {"name": "depletion_power", "symbol": "\u0394P", "description": "Depletion beam power error (%)", "nominal": 0, "perturbed": 10.0},
            {"name": "donut_alignment", "symbol": "\u0394r", "description": "Donut beam alignment error (nm)", "nominal": 0, "perturbed": 10},
            {"name": "saturation_intensity", "symbol": "\u0394I_s", "description": "Saturation intensity error (%)", "nominal": 0, "perturbed": 8.0},
        ],
    },

    "tirf": {
        "display_name": "TIRF",
        "full_name": "TIRF Microscopy",
        "parent_modality": "tirf",
        "category": "microscopy",
        "spec_notation": "C(PSF_TIRF) \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF_TIRF", "label": "Evanescent-Field PSF"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "EMCCD / sCMOS"},
        ],
        "mismatch_params": [
            {"name": "incidence_angle", "symbol": "\u0394\u03b8_i", "description": "Incidence angle error (deg)", "nominal": 0, "perturbed": 0.3},
            {"name": "penetration_depth", "symbol": "\u0394d", "description": "Penetration depth error (nm)", "nominal": 0, "perturbed": 20},
            {"name": "refractive_index", "symbol": "\u0394n", "description": "Coverslip refractive index error", "nominal": 1.515, "perturbed": 1.52},
        ],
    },

    "fundus": {
        "display_name": "Fundus",
        "full_name": "Fundus Camera",
        "parent_modality": "fundus",
        "category": "clinical_optics",
        "spec_notation": "C(PSF_optic) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF_optic", "label": "Ophthalmic PSF"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "CCD Camera"},
        ],
        "mismatch_params": [
            {"name": "pupil_dilation", "symbol": "\u0394d_p", "description": "Pupil dilation error (mm)", "nominal": 0, "perturbed": 0.5},
            {"name": "focus", "symbol": "\u0394f", "description": "Focus error (diopters)", "nominal": 0, "perturbed": 0.25},
            {"name": "vignetting", "symbol": "\u0394v", "description": "Vignetting uniformity error (%)", "nominal": 0, "perturbed": 5.0},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  MICROSCOPY — Structured Illumination  (2 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "sim": {
        "display_name": "SIM",
        "full_name": "Structured Illumination Microscopy",
        "parent_modality": "sim",
        "category": "microscopy",
        "spec_notation": "S(grating) \u2192 C(PSF) \u2192 \u03a3_\u03c6 \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "S", "params": "grating", "label": "Sinusoidal Illumination"},
            {"primitive": "C", "params": "PSF", "label": "PSF Convolution"},
            {"primitive": "Sigma", "params": "\u03c6", "label": "Phase-Shift Sum"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "sCMOS Camera"},
        ],
        "mismatch_params": [
            {"name": "pattern_phase", "symbol": "\u0394\u03c6", "description": "Pattern phase error (rad)", "nominal": 0, "perturbed": 0.05},
            {"name": "pattern_freq", "symbol": "\u0394k", "description": "Pattern frequency error (%)", "nominal": 0, "perturbed": 1.0},
            {"name": "modulation_depth", "symbol": "\u0394m", "description": "Modulation depth error (%)", "nominal": 0, "perturbed": 5.0},
        ],
    },

    "fpm": {
        "display_name": "FPM",
        "full_name": "Fourier Ptychographic Microscopy",
        "parent_modality": "fpm",
        "category": "microscopy",
        "spec_notation": "S(LED array) \u2192 C(PSF_NA) \u2192 \u03a3_\u03b8 \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "S", "params": "LED", "label": "LED Array Illumination"},
            {"primitive": "C", "params": "PSF_NA", "label": "Low-NA PSF"},
            {"primitive": "Sigma", "params": "\u03b8", "label": "Angular Sum"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Camera"},
        ],
        "mismatch_params": [
            {"name": "led_position", "symbol": "\u0394r_LED", "description": "LED position error (mm)", "nominal": 0, "perturbed": 0.1},
            {"name": "na_error", "symbol": "\u0394NA", "description": "Numerical aperture error", "nominal": 0.1, "perturbed": 0.105},
            {"name": "defocus", "symbol": "\u0394z", "description": "Defocus error (\u03bcm)", "nominal": 0, "perturbed": 2.0},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  MICROSCOPY — Special  (3 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "flim": {
        "display_name": "FLIM",
        "full_name": "Fluorescence Lifetime Imaging",
        "parent_modality": "flim",
        "category": "microscopy",
        "spec_notation": "C(PSF) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF", "label": "PSF Convolution"},
            {"primitive": "Sigma", "params": "t", "label": "Time-Gate Sum"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "TCSPC Detector"},
        ],
        "mismatch_params": [
            {"name": "irf_width", "symbol": "\u0394IRF", "description": "Instrument response width error (ps)", "nominal": 0, "perturbed": 20},
            {"name": "time_bin", "symbol": "\u0394\u03c4_b", "description": "Time bin error (ps)", "nominal": 0, "perturbed": 5},
            {"name": "afterpulsing", "symbol": "p_ap", "description": "Afterpulsing probability", "nominal": 0, "perturbed": 0.005},
        ],
    },

    "palm_storm": {
        "display_name": "PALM/STORM",
        "full_name": "PALM/STORM Single-Molecule Localization",
        "parent_modality": "palm_storm",
        "category": "microscopy",
        "spec_notation": "C(PSF) \u2192 D(g, \u03b7\u2083)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF", "label": "Single-Molecule PSF"},
            {"primitive": "D", "params": "g, \u03b7\u2083", "label": "EMCCD / sCMOS"},
        ],
        "mismatch_params": [
            {"name": "psf_model", "symbol": "\u0394PSF", "description": "PSF model error (Gaussian vs Airy)", "nominal": 0, "perturbed": 5.0},
            {"name": "emitter_density", "symbol": "\u0394\u03c1", "description": "Emitter density error (%)", "nominal": 0, "perturbed": 20.0},
            {"name": "drift", "symbol": "\u0394r", "description": "Stage drift (nm/frame)", "nominal": 0, "perturbed": 0.5},
        ],
    },

    "polarization": {
        "display_name": "Polarization",
        "full_name": "Polarization Microscopy",
        "parent_modality": "polarization",
        "category": "microscopy",
        "spec_notation": "M(polarizer) \u2192 C(PSF) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "M", "params": "polarizer", "label": "Polarizer / Analyzer"},
            {"primitive": "C", "params": "PSF", "label": "PSF Convolution"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Camera"},
        ],
        "mismatch_params": [
            {"name": "extinction_ratio", "symbol": "\u0394ER", "description": "Extinction ratio error (dB)", "nominal": 0, "perturbed": 0.5},
            {"name": "retardance", "symbol": "\u0394\u03b4", "description": "Retardance error (nm)", "nominal": 0, "perturbed": 2.0},
            {"name": "alignment", "symbol": "\u0394\u03b8", "description": "Polarizer alignment error (deg)", "nominal": 0, "perturbed": 0.5},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  ELECTRON MICROSCOPY  (8 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "sem": {
        "display_name": "SEM",
        "full_name": "Scanning Electron Microscopy",
        "parent_modality": "sem",
        "category": "electron_microscopy",
        "spec_notation": "P(e\u207b beam) \u2192 C(probe) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "e\u207b", "label": "Electron Beam"},
            {"primitive": "C", "params": "probe", "label": "Probe Scanning"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "SE / BSE Detector"},
        ],
        "mismatch_params": [
            {"name": "beam_energy", "symbol": "\u0394E", "description": "Beam energy error (keV)", "nominal": 0, "perturbed": 0.1},
            {"name": "stigmatism", "symbol": "\u0394A_s", "description": "Astigmatism (nm)", "nominal": 0, "perturbed": 5.0},
            {"name": "working_distance", "symbol": "\u0394WD", "description": "Working distance error (mm)", "nominal": 0, "perturbed": 0.1},
        ],
    },

    "tem": {
        "display_name": "TEM",
        "full_name": "Transmission Electron Microscopy",
        "parent_modality": "tem",
        "category": "electron_microscopy",
        "spec_notation": "P(e\u207b) \u2192 C(CTF) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "e\u207b", "label": "Electron Wave"},
            {"primitive": "C", "params": "CTF", "label": "Contrast Transfer Function"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Direct Electron Detector"},
        ],
        "mismatch_params": [
            {"name": "defocus", "symbol": "\u0394f", "description": "Defocus error (nm)", "nominal": 0, "perturbed": 50},
            {"name": "Cs", "symbol": "\u0394C_s", "description": "Spherical aberration error (mm)", "nominal": 0, "perturbed": 0.01},
            {"name": "beam_tilt", "symbol": "\u0394\u03b8", "description": "Beam tilt error (mrad)", "nominal": 0, "perturbed": 0.5},
        ],
    },

    "stem": {
        "display_name": "STEM",
        "full_name": "Scanning TEM",
        "parent_modality": "stem",
        "category": "electron_microscopy",
        "spec_notation": "P(e\u207b) \u2192 C(probe) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "e\u207b", "label": "Electron Probe"},
            {"primitive": "C", "params": "probe", "label": "Probe Formation"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "ADF / BF Detector"},
        ],
        "mismatch_params": [
            {"name": "probe_size", "symbol": "\u0394d_p", "description": "Probe size error (\u00c5)", "nominal": 0, "perturbed": 0.1},
            {"name": "convergence_angle", "symbol": "\u0394\u03b1", "description": "Convergence semi-angle error (mrad)", "nominal": 0, "perturbed": 0.5},
            {"name": "scan_distortion", "symbol": "\u0394s", "description": "Scan distortion (%)", "nominal": 0, "perturbed": 0.5},
        ],
    },

    "electron_tomography": {
        "display_name": "Electron Tomo",
        "full_name": "Electron Tomography",
        "parent_modality": "electron_tomography",
        "category": "electron_microscopy",
        "spec_notation": "R(\u03b8) \u2192 P(e\u207b) \u2192 \u03a0(proj) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "R", "params": "\u03b8", "label": "Sample Tilt"},
            {"primitive": "P", "params": "e\u207b", "label": "Electron Wave"},
            {"primitive": "Pi", "params": "proj", "label": "Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Detector"},
        ],
        "mismatch_params": [
            {"name": "tilt_angle", "symbol": "\u0394\u03b8", "description": "Tilt angle error (deg)", "nominal": 0, "perturbed": 0.5},
            {"name": "tilt_axis", "symbol": "\u0394\u03c6", "description": "Tilt axis misalignment (deg)", "nominal": 0, "perturbed": 0.3},
            {"name": "defocus_gradient", "symbol": "\u0394f'", "description": "Defocus gradient (nm/\u03bcm)", "nominal": 0, "perturbed": 10},
        ],
    },

    "electron_diffraction": {
        "display_name": "4D-STEM",
        "full_name": "4D-STEM Electron Diffraction",
        "parent_modality": "electron_diffraction",
        "category": "electron_microscopy",
        "spec_notation": "P(e\u207b) \u2192 F(diffraction) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "e\u207b", "label": "Convergent Beam"},
            {"primitive": "F", "params": "diffraction", "label": "Diffraction Pattern"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Pixelated Detector"},
        ],
        "mismatch_params": [
            {"name": "camera_length", "symbol": "\u0394L", "description": "Camera length error (%)", "nominal": 0, "perturbed": 2.0},
            {"name": "center_offset", "symbol": "\u0394c", "description": "Diffraction center offset (pixels)", "nominal": 0, "perturbed": 1.0},
            {"name": "elliptical_distortion", "symbol": "\u03b5", "description": "Elliptical distortion", "nominal": 0, "perturbed": 0.005},
        ],
    },

    "electron_holography": {
        "display_name": "Electron Holography",
        "full_name": "Electron Holography",
        "parent_modality": "electron_holography",
        "category": "electron_microscopy",
        "spec_notation": "P(e\u207b) \u2192 \u03a3(interference) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "e\u207b", "label": "Electron Biprism"},
            {"primitive": "Sigma", "params": "interference", "label": "Interference Sum"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "CCD Camera"},
        ],
        "mismatch_params": [
            {"name": "biprism_voltage", "symbol": "\u0394V_b", "description": "Biprism voltage error (V)", "nominal": 0, "perturbed": 2.0},
            {"name": "fringe_spacing", "symbol": "\u0394d_f", "description": "Fringe spacing error (nm)", "nominal": 0, "perturbed": 0.1},
            {"name": "partial_coherence", "symbol": "\u0394\u03bc", "description": "Spatial coherence error (%)", "nominal": 0, "perturbed": 5.0},
        ],
    },

    "ebsd": {
        "display_name": "EBSD",
        "full_name": "Electron Backscatter Diffraction",
        "parent_modality": "ebsd",
        "category": "electron_microscopy",
        "spec_notation": "P(e\u207b) \u2192 \u03a0(backscatter) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "e\u207b", "label": "Electron Beam"},
            {"primitive": "Pi", "params": "backscatter", "label": "Kikuchi Pattern Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Phosphor Screen + Camera"},
        ],
        "mismatch_params": [
            {"name": "pattern_center", "symbol": "\u0394PC", "description": "Pattern center error (pixels)", "nominal": 0, "perturbed": 2.0},
            {"name": "sample_tilt", "symbol": "\u0394\u03b8", "description": "Sample tilt error (deg)", "nominal": 70, "perturbed": 70.5},
            {"name": "detector_distance", "symbol": "\u0394DD", "description": "Detector distance error (mm)", "nominal": 0, "perturbed": 0.5},
        ],
    },

    "eels": {
        "display_name": "EELS",
        "full_name": "Electron Energy Loss Spectroscopy",
        "parent_modality": "eels",
        "category": "electron_microscopy",
        "spec_notation": "P(e\u207b) \u2192 \u039b(energy) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "e\u207b", "label": "Electron Beam"},
            {"primitive": "Lambda", "params": "energy", "label": "Energy Disperser"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "CCD Spectrometer"},
        ],
        "mismatch_params": [
            {"name": "energy_dispersion", "symbol": "\u0394D_E", "description": "Energy dispersion error (eV/channel)", "nominal": 0, "perturbed": 0.002},
            {"name": "zero_loss_shift", "symbol": "\u0394E_0", "description": "Zero-loss peak shift (eV)", "nominal": 0, "perturbed": 0.3},
            {"name": "aberration", "symbol": "\u0394C_c", "description": "Chromatic aberration error (%)", "nominal": 0, "perturbed": 2.0},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  CLINICAL OPTICS  (4 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "oct": {
        "display_name": "OCT",
        "full_name": "Optical Coherence Tomography",
        "parent_modality": "oct",
        "category": "clinical_optics",
        "spec_notation": "P(low-coherence) \u2192 \u03a3(interference) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "low-coherence", "label": "Low-Coherence Source"},
            {"primitive": "Sigma", "params": "interference", "label": "Interferometric Sum"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Spectrometer / Balanced Detector"},
        ],
        "mismatch_params": [
            {"name": "dispersion", "symbol": "\u0394GVD", "description": "Dispersion mismatch (fs\u00b2)", "nominal": 0, "perturbed": 200},
            {"name": "reference_delay", "symbol": "\u0394z_r", "description": "Reference delay error (\u03bcm)", "nominal": 0, "perturbed": 5.0},
            {"name": "spectral_roll_off", "symbol": "\u0394R", "description": "Spectral roll-off error (dB/mm)", "nominal": 0, "perturbed": 1.0},
        ],
    },

    "octa": {
        "display_name": "OCTA",
        "full_name": "OCT Angiography",
        "parent_modality": "octa",
        "category": "clinical_optics",
        "spec_notation": "P(low-coherence) \u2192 \u03a3(interference) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "low-coherence", "label": "Low-Coherence Source"},
            {"primitive": "Sigma", "params": "interference", "label": "Interferometric Sum"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Spectrometer"},
        ],
        "mismatch_params": [
            {"name": "inter_bscan_time", "symbol": "\u0394T", "description": "Inter-B-scan time error (ms)", "nominal": 0, "perturbed": 0.5},
            {"name": "bulk_motion", "symbol": "\u0394v_b", "description": "Bulk motion artifact (mm/s)", "nominal": 0, "perturbed": 0.2},
            {"name": "decorrelation_threshold", "symbol": "\u0394D_th", "description": "Decorrelation threshold error", "nominal": 0.5, "perturbed": 0.55},
        ],
    },

    "endoscopy": {
        "display_name": "Endoscopy",
        "full_name": "Fiber Bundle Endoscopy",
        "parent_modality": "endoscopy",
        "category": "clinical_optics",
        "spec_notation": "C(PSF_fiber) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF_fiber", "label": "Fiber Bundle PSF"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Camera"},
        ],
        "mismatch_params": [
            {"name": "fiber_coupling", "symbol": "\u0394\u03b7_f", "description": "Fiber coupling efficiency error (%)", "nominal": 0, "perturbed": 5.0},
            {"name": "core_spacing", "symbol": "\u0394d", "description": "Core spacing error (\u03bcm)", "nominal": 0, "perturbed": 0.5},
            {"name": "bending_loss", "symbol": "\u0394\u03b1_b", "description": "Bending loss error (dB)", "nominal": 0, "perturbed": 0.3},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  COMPUTATIONAL  (2 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "light_field": {
        "display_name": "Light Field",
        "full_name": "Light Field Imaging",
        "parent_modality": "light_field",
        "category": "computational",
        "spec_notation": "\u03a0(micro-lens) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "Pi", "params": "micro-lens", "label": "Micro-Lens Array Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Sensor Array"},
        ],
        "mismatch_params": [
            {"name": "microlens_pitch", "symbol": "\u0394p", "description": "Micro-lens pitch error (\u03bcm)", "nominal": 0, "perturbed": 0.5},
            {"name": "main_lens_f", "symbol": "\u0394f", "description": "Main lens focal length error (mm)", "nominal": 0, "perturbed": 0.1},
            {"name": "vignetting", "symbol": "\u0394v", "description": "Vignetting error (%)", "nominal": 0, "perturbed": 3.0},
        ],
    },

    "integral": {
        "display_name": "Integral",
        "full_name": "Integral Photography",
        "parent_modality": "integral",
        "category": "computational",
        "spec_notation": "\u03a0(lens-array) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "Pi", "params": "lens-array", "label": "Lens Array Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Sensor"},
        ],
        "mismatch_params": [
            {"name": "lens_pitch", "symbol": "\u0394p", "description": "Lens pitch error (\u03bcm)", "nominal": 0, "perturbed": 1.0},
            {"name": "gap_distance", "symbol": "\u0394d", "description": "Lens-to-sensor gap error (\u03bcm)", "nominal": 0, "perturbed": 5.0},
            {"name": "aberration", "symbol": "\u0394W", "description": "Lens aberration (waves)", "nominal": 0, "perturbed": 0.1},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  COMPUTATIONAL PHOTOGRAPHY  (2 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "lensless": {
        "display_name": "Lensless",
        "full_name": "Lensless (Diffuser Camera) Imaging",
        "parent_modality": "lensless",
        "category": "computational_photography",
        "spec_notation": "P(diffuser) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "diffuser", "label": "Diffuser Propagation"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Bare Sensor"},
        ],
        "mismatch_params": [
            {"name": "diffuser_psf", "symbol": "\u0394PSF", "description": "Diffuser PSF calibration error (%)", "nominal": 0, "perturbed": 5.0},
            {"name": "sensor_distance", "symbol": "\u0394d", "description": "Diffuser-sensor distance error (mm)", "nominal": 0, "perturbed": 0.2},
            {"name": "wavelength", "symbol": "\u0394\u03bb", "description": "Wavelength mismatch (nm)", "nominal": 0, "perturbed": 5.0},
        ],
    },

    "panorama": {
        "display_name": "Panorama",
        "full_name": "Panorama Multi-Focus Fusion",
        "parent_modality": "panorama",
        "category": "computational_photography",
        "spec_notation": "C(PSF_focus) \u2192 \u03a3_f \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "C", "params": "PSF_focus", "label": "Depth-Dependent PSF"},
            {"primitive": "Sigma", "params": "f", "label": "Focus Stack Sum"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Camera"},
        ],
        "mismatch_params": [
            {"name": "focus_step", "symbol": "\u0394f", "description": "Focus step error (\u03bcm)", "nominal": 0, "perturbed": 2.0},
            {"name": "registration", "symbol": "\u0394r", "description": "Inter-frame registration error (pixels)", "nominal": 0, "perturbed": 0.5},
            {"name": "exposure_variation", "symbol": "\u0394E", "description": "Exposure variation (%)", "nominal": 0, "perturbed": 3.0},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  NEURAL RENDERING  (2 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "nerf": {
        "display_name": "NeRF",
        "full_name": "Neural Radiance Fields",
        "parent_modality": "nerf",
        "category": "neural_rendering",
        "spec_notation": "\u03a0(ray) \u2192 \u03a3(volume) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "Pi", "params": "ray", "label": "Ray Casting"},
            {"primitive": "Sigma", "params": "volume", "label": "Volume Rendering Integral"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Camera"},
        ],
        "mismatch_params": [
            {"name": "camera_pose", "symbol": "\u0394T", "description": "Camera pose error (mm / deg)", "nominal": 0, "perturbed": 1.0},
            {"name": "focal_length", "symbol": "\u0394f", "description": "Focal length error (pixels)", "nominal": 0, "perturbed": 5.0},
            {"name": "distortion", "symbol": "\u0394k", "description": "Radial distortion coefficient error", "nominal": 0, "perturbed": 0.01},
            {"name": "exposure", "symbol": "\u0394E", "description": "Exposure variation (%)", "nominal": 0, "perturbed": 10.0},
        ],
    },

    "gaussian_splatting": {
        "display_name": "3DGS",
        "full_name": "3D Gaussian Splatting",
        "parent_modality": "gaussian_splatting",
        "category": "neural_rendering",
        "spec_notation": "\u03a0(splat) \u2192 \u03a3(alpha) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "Pi", "params": "splat", "label": "Gaussian Splatting"},
            {"primitive": "Sigma", "params": "alpha", "label": "Alpha Compositing"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Camera"},
        ],
        "mismatch_params": [
            {"name": "camera_pose", "symbol": "\u0394T", "description": "Camera pose error (mm / deg)", "nominal": 0, "perturbed": 1.0},
            {"name": "focal_length", "symbol": "\u0394f", "description": "Focal length error (pixels)", "nominal": 0, "perturbed": 5.0},
            {"name": "point_cloud_init", "symbol": "\u0394P", "description": "Initial point cloud noise (mm)", "nominal": 0, "perturbed": 2.0},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  DEPTH IMAGING  (3 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "tof_camera": {
        "display_name": "ToF Camera",
        "full_name": "Time-of-Flight Depth Camera",
        "parent_modality": "tof_camera",
        "category": "depth_imaging",
        "spec_notation": "P(modulated) \u2192 \u03a3(correlation) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "modulated", "label": "Modulated Light"},
            {"primitive": "Sigma", "params": "correlation", "label": "Correlation Integration"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "ToF Sensor"},
        ],
        "mismatch_params": [
            {"name": "modulation_freq", "symbol": "\u0394f_m", "description": "Modulation frequency error (MHz)", "nominal": 20, "perturbed": 20.1},
            {"name": "multipath", "symbol": "I_mp", "description": "Multipath interference intensity (%)", "nominal": 0, "perturbed": 5.0},
            {"name": "phase_nonlinearity", "symbol": "\u0394\u03c6", "description": "Phase nonlinearity (deg)", "nominal": 0, "perturbed": 2.0},
        ],
    },

    "structured_light": {
        "display_name": "Structured Light",
        "full_name": "Structured-Light Depth Camera",
        "parent_modality": "structured_light",
        "category": "depth_imaging",
        "spec_notation": "S(pattern) \u2192 \u03a0(triangulation) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "S", "params": "pattern", "label": "Projected Pattern"},
            {"primitive": "Pi", "params": "triangulation", "label": "Triangulation"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "IR Camera"},
        ],
        "mismatch_params": [
            {"name": "baseline", "symbol": "\u0394b", "description": "Projector-camera baseline error (mm)", "nominal": 0, "perturbed": 0.5},
            {"name": "pattern_distortion", "symbol": "\u0394P", "description": "Pattern distortion (%)", "nominal": 0, "perturbed": 1.0},
            {"name": "ambient_ir", "symbol": "I_amb", "description": "Ambient IR interference (%)", "nominal": 0, "perturbed": 3.0},
        ],
    },

    "lidar": {
        "display_name": "LiDAR",
        "full_name": "LiDAR Scanner",
        "parent_modality": "lidar",
        "category": "depth_imaging",
        "spec_notation": "P(pulsed) \u2192 \u03a3(return) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "P", "params": "pulsed", "label": "Pulsed Laser"},
            {"primitive": "Sigma", "params": "return", "label": "Return Signal Integration"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "SPAD / APD"},
        ],
        "mismatch_params": [
            {"name": "timing_jitter", "symbol": "\u0394t", "description": "Timing jitter (ps)", "nominal": 0, "perturbed": 50},
            {"name": "beam_divergence", "symbol": "\u0394\u03b8", "description": "Beam divergence error (mrad)", "nominal": 0, "perturbed": 0.1},
            {"name": "range_walk", "symbol": "\u0394R", "description": "Range walk error (cm)", "nominal": 0, "perturbed": 1.0},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  REMOTE SENSING  (2 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "sar": {
        "display_name": "SAR",
        "full_name": "Synthetic Aperture Radar",
        "parent_modality": "sar",
        "category": "remote_sensing",
        "spec_notation": "F(azimuth\u00d7range) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "F", "params": "azimuth\u00d7range", "label": "Range-Doppler Sampling"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Radar Receiver"},
        ],
        "mismatch_params": [
            {"name": "motion_error", "symbol": "\u0394r_a", "description": "Platform motion error (cm)", "nominal": 0, "perturbed": 2.0},
            {"name": "phase_error", "symbol": "\u0394\u03c6", "description": "Autofocus phase error (rad)", "nominal": 0, "perturbed": 0.3},
            {"name": "range_cell_migration", "symbol": "\u0394RCM", "description": "Range cell migration error (cells)", "nominal": 0, "perturbed": 0.5},
        ],
    },

    "sonar": {
        "display_name": "Sonar",
        "full_name": "Sonar Imaging",
        "parent_modality": "sonar",
        "category": "remote_sensing",
        "spec_notation": "P(acoustic) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2082)",
        "spec_dag": [
            {"primitive": "P", "params": "acoustic", "label": "Acoustic Propagation"},
            {"primitive": "Sigma", "params": "t", "label": "Echo Integration"},
            {"primitive": "D", "params": "g, \u03b7\u2082", "label": "Hydrophone Array"},
        ],
        "mismatch_params": [
            {"name": "sound_speed_profile", "symbol": "\u0394c(z)", "description": "Sound speed profile error (m/s)", "nominal": 0, "perturbed": 5.0},
            {"name": "multipath", "symbol": "n_mp", "description": "Number of unmodeled multipaths", "nominal": 0, "perturbed": 2},
            {"name": "array_calibration", "symbol": "\u0394a", "description": "Array element calibration error (%)", "nominal": 0, "perturbed": 3.0},
        ],
    },

    # ══════════════════════════════════════════════════════════════════════════
    #  PARTICLE IMAGING  (3 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "neutron_tomo": {
        "display_name": "Neutron Tomo",
        "full_name": "Neutron Radiography / Tomography",
        "parent_modality": "neutron_tomo",
        "category": "particle_imaging",
        "spec_notation": "R(\u03b8) \u2192 \u03a0(neutron) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "R", "params": "\u03b8", "label": "Sample Rotation"},
            {"primitive": "Pi", "params": "neutron", "label": "Neutron Attenuation Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Scintillator + CCD"},
        ],
        "mismatch_params": [
            {"name": "beam_spectrum", "symbol": "\u0394E", "description": "Beam energy spectrum error (%)", "nominal": 0, "perturbed": 3.0},
            {"name": "scatter_correction", "symbol": "\u0394s", "description": "Scatter correction error (%)", "nominal": 0, "perturbed": 5.0},
            {"name": "rotation_offset", "symbol": "\u0394\u03b8", "description": "Rotation center offset (pixels)", "nominal": 0, "perturbed": 1.0},
        ],
    },

    "proton_radiography": {
        "display_name": "Proton Radiography",
        "full_name": "Proton Radiography",
        "parent_modality": "proton_radiography",
        "category": "particle_imaging",
        "spec_notation": "\u03a0(proton) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "Pi", "params": "proton", "label": "Proton Transmission"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Tracking Detector"},
        ],
        "mismatch_params": [
            {"name": "energy_loss", "symbol": "\u0394S", "description": "Stopping power error (%)", "nominal": 0, "perturbed": 2.0},
            {"name": "scattering", "symbol": "\u0394\u03b8_MCS", "description": "Multiple Coulomb scattering error (%)", "nominal": 0, "perturbed": 5.0},
            {"name": "range_straggling", "symbol": "\u0394R", "description": "Range straggling error (%)", "nominal": 0, "perturbed": 3.0},
        ],
    },

    "muon_tomo": {
        "display_name": "Muon Tomo",
        "full_name": "Muon Tomography",
        "parent_modality": "muon_tomo",
        "category": "particle_imaging",
        "spec_notation": "R(\u03b8_cosmic) \u2192 \u03a0(muon) \u2192 D(g, \u03b7\u2081)",
        "spec_dag": [
            {"primitive": "R", "params": "\u03b8_cosmic", "label": "Cosmic Muon Incidence"},
            {"primitive": "Pi", "params": "muon", "label": "Muon Scattering Projection"},
            {"primitive": "D", "params": "g, \u03b7\u2081", "label": "Drift Tube Tracker"},
        ],
        "mismatch_params": [
            {"name": "angular_resolution", "symbol": "\u0394\u03b8", "description": "Angular resolution error (mrad)", "nominal": 0, "perturbed": 2.0},
            {"name": "momentum_estimate", "symbol": "\u0394p", "description": "Muon momentum error (%)", "nominal": 0, "perturbed": 10.0},
            {"name": "detector_efficiency", "symbol": "\u0394\u03b5", "description": "Detector efficiency error (%)", "nominal": 0, "perturbed": 3.0},
        ],
    },
    # ══════════════════════════════════════════════════════════════════════════
    #  MULTIMODALITY  (5 variants)
    # ══════════════════════════════════════════════════════════════════════════

    "pet_ct": {
        "display_name": "PET/CT",
        "full_name": "Positron Emission Tomography / Computed Tomography",
        "parent_modality": "pet_ct",
        "category": "medical",
        "spec_notation": "R(θ) → D(μ_ct) → Π(LOR) → D(g, η)",
        "spec_dag": [
            {"primitive": "R", "params": "θ", "label": "Radon Transform"},
            {"primitive": "D", "params": "μ_ct", "label": "CT Attenuation Map"},
            {"primitive": "Pi", "params": "LOR", "label": "PET Line of Response"},
            {"primitive": "D", "params": "g, η", "label": "Detector + Noise"},
        ],
        "mismatch_params": [
            {"name": "ct_registration_shift", "symbol": "Δs", "description": "CT-PET registration error (pixels)", "nominal": 0, "perturbed": 4.0},
            {"name": "hu_to_mu_scale", "symbol": "Δμ", "description": "HU-to-μ calibration error (%)", "nominal": 0, "perturbed": 10.0},
            {"name": "scatter_fraction", "symbol": "f_s", "description": "Scatter fraction", "nominal": 0, "perturbed": 0.15},
        ],
    },

    "pet_mr": {
        "display_name": "PET/MR",
        "full_name": "Positron Emission Tomography / Magnetic Resonance",
        "parent_modality": "pet_mr",
        "category": "medical",
        "spec_notation": "F(k) → S(coil) → D(μ_mr) → R(θ) → D(g, η)",
        "spec_dag": [
            {"primitive": "F", "params": "k", "label": "MR k-Space (FFT)"},
            {"primitive": "S", "params": "coil", "label": "Coil Sensitivities"},
            {"primitive": "D", "params": "μ_mr", "label": "MR-Based Attenuation"},
            {"primitive": "R", "params": "θ", "label": "PET Radon Transform"},
            {"primitive": "D", "params": "g, η", "label": "Detector + Noise"},
        ],
        "mismatch_params": [
            {"name": "mr_attenuation_error", "symbol": "Δμ_mr", "description": "MR attenuation estimation error (%)", "nominal": 0, "perturbed": 25.0},
            {"name": "motion_shift", "symbol": "Δs", "description": "Inter-modality motion (pixels)", "nominal": 0, "perturbed": 6.0},
            {"name": "b0_inhomogeneity", "symbol": "ΔB0", "description": "B0 inhomogeneity (Hz)", "nominal": 0, "perturbed": 40.0},
            {"name": "pet_scatter_fraction", "symbol": "f_s", "description": "PET scatter fraction", "nominal": 0, "perturbed": 0.25},
        ],
    },

    "spect_ct": {
        "display_name": "SPECT/CT",
        "full_name": "Single Photon Emission CT / Computed Tomography",
        "parent_modality": "spect_ct",
        "category": "medical",
        "spec_notation": "R(θ) → D(μ_ct) → H(coll) → D(g, η)",
        "spec_dag": [
            {"primitive": "R", "params": "θ", "label": "Radon Transform"},
            {"primitive": "D", "params": "μ_ct", "label": "CT Attenuation Map"},
            {"primitive": "H", "params": "coll", "label": "Collimator PSF"},
            {"primitive": "D", "params": "g, η", "label": "Detector + Noise"},
        ],
        "mismatch_params": [
            {"name": "ct_registration_shift", "symbol": "Δs", "description": "CT-SPECT registration error (pixels)", "nominal": 0, "perturbed": 5.0},
            {"name": "hu_to_mu_scale", "symbol": "Δμ", "description": "HU-to-μ calibration error (%)", "nominal": 0, "perturbed": 12.0},
            {"name": "scatter_fraction", "symbol": "f_s", "description": "Scatter fraction", "nominal": 0, "perturbed": 0.35},
            {"name": "collimator_blur", "symbol": "σ_c", "description": "Collimator blur FWHM (pixels)", "nominal": 2.5, "perturbed": 6.0},
        ],
    },

    "spectral_ct": {
        "display_name": "Spectral CT",
        "full_name": "Dual-Energy / Spectral Computed Tomography",
        "parent_modality": "spectral_ct",
        "category": "medical",
        "spec_notation": "R(θ) → A(E_low, E_high) → Σ_mat → D(g, η)",
        "spec_dag": [
            {"primitive": "R", "params": "θ", "label": "Radon Transform"},
            {"primitive": "A", "params": "E_low, E_high", "label": "Energy-Dependent Attenuation"},
            {"primitive": "Sigma", "params": "mat", "label": "Material Decomposition"},
            {"primitive": "D", "params": "g, η", "label": "Detector + Noise"},
        ],
        "mismatch_params": [
            {"name": "energy_calibration_error", "symbol": "ΔE", "description": "Energy calibration error (keV)", "nominal": 0, "perturbed": 4.0},
            {"name": "scatter_fraction", "symbol": "f_s", "description": "Scatter fraction", "nominal": 0, "perturbed": 0.20},
            {"name": "detector_crosstalk", "symbol": "ε_xt", "description": "Cross-energy detector leakage", "nominal": 0, "perturbed": 0.10},
            {"name": "beam_hardening", "symbol": "β_bh", "description": "Beam hardening coefficient", "nominal": 0, "perturbed": 0.20},
        ],
    },

    "industrial_ct": {
        "display_name": "Industrial CT",
        "full_name": "Industrial Computed Tomography",
        "parent_modality": "industrial_ct",
        "category": "medical",
        "spec_notation": "R(θ) → B(poly) → S(scatter) → D(g, η)",
        "spec_dag": [
            {"primitive": "R", "params": "θ", "label": "Radon Transform"},
            {"primitive": "B", "params": "poly", "label": "Beam Hardening (Polychromatic)"},
            {"primitive": "S", "params": "scatter", "label": "Scatter Contribution"},
            {"primitive": "D", "params": "g, η", "label": "Detector + Noise"},
        ],
        "mismatch_params": [
            {"name": "beam_hardening_order", "symbol": "β_bh", "description": "Beam hardening polynomial coefficient", "nominal": 0, "perturbed": 0.5},
            {"name": "scatter_fraction", "symbol": "f_s", "description": "Scatter fraction", "nominal": 0, "perturbed": 0.15},
            {"name": "source_blur", "symbol": "σ_src", "description": "Source focal spot blur (pixels)", "nominal": 0, "perturbed": 3.0},
            {"name": "detector_efficiency", "symbol": "η_det", "description": "Detector efficiency variation", "nominal": 1.0, "perturbed": 0.85},
        ],
    },
}

# fmt: on
