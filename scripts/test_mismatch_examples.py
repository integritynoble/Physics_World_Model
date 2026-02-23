#!/usr/bin/env python3
"""Mismatch examples & modality introductions for all 64 PWM imaging modalities.

For each modality, this script:
  1. Builds the CORRECT operator (via _build_operator_by_id or _try_build_graph_operator)
  2. Builds the WIDEFIELD FALLBACK (WidefieldOperator, sigma=2.0)
  3. Runs forward pass on both and compares outputs (Example 1)
  4. Runs round-trip (forward -> adjoint) on both and compares (Example 2)
  5. Optionally runs domain/linearity checks (Example 3)
  6. Generates docs/modality_mismatch_guide.md with per-modality introductions

Before commit 02427aa, ~46 modalities fell through to the widefield Gaussian blur
fallback — physically wrong for nearly every non-widefield modality. This script
documents exactly what goes wrong for each one.

Usage:
    .venv/bin/python3 scripts/test_mismatch_examples.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# All 64 modalities (same canonical list as test_all_forward_models.py)
# ---------------------------------------------------------------------------
ALL_MODALITIES = [
    # Microscopy
    "widefield", "widefield_lowdose", "confocal_livecell", "confocal_3d",
    "sim", "lightsheet",
    # Compressive / Spectral
    "cassi", "spc", "cacti", "matrix",
    # Tomography / MRI
    "ct", "mri",
    # Phase / Holography / Coherent
    "ptychography", "holography", "phase_retrieval", "fpm",
    # Rendering / 3D
    "nerf", "gaussian_splatting",
    # Lensless / Computational
    "lensless", "panorama", "light_field",
    # Biomedical
    "dot", "photoacoustic", "oct", "flim", "integral",
    # New dedicated operators
    "xray_radiography", "ultrasound", "pet", "spect",
    "sem", "tem", "electron_tomography",
    # Additional v2 modalities (graph-only)
    "stem", "fluoroscopy", "mammography", "dexa", "cbct",
    "angiography", "doppler_ultrasound", "elastography",
    "fmri", "mrs", "diffusion_mri",
    "two_photon", "sted", "palm_storm", "tirf",
    "polarization", "endoscopy", "fundus", "octa",
    "tof_camera", "lidar", "structured_light",
    "sar", "sonar",
    "electron_diffraction", "ebsd", "eels", "electron_holography",
    "neutron_tomo", "proton_radiography", "muon_tomo",
]

DEDICATED_MODALITIES = {
    "widefield", "sim", "cassi", "spc", "cacti",
    "lensless", "lightsheet", "ct", "mri", "ptychography", "holography",
    "nerf", "gaussian_splatting", "oct", "light_field",
    "photoacoustic", "fpm", "flim", "dot", "integral",
    "phase_retrieval", "cdi",
    "ultrasound", "sem", "tem", "electron_tomography",
    "pet", "spect", "xray_radiography",
    "matrix",
}

MODALITY_TO_DEDICATED = {
    "widefield_lowdose": "widefield",
    "confocal_livecell": "confocal",
    "confocal_3d": "confocal",
}

# ---------------------------------------------------------------------------
# Modality specification registry
# ---------------------------------------------------------------------------


@dataclass
class ModalitySpec:
    modality: str
    category: str
    display_name: str
    physics_intro: str
    forward_equation: str
    correct_x_shape: Tuple[int, ...]  # operator build shape (test dims)
    correct_y_shape: Tuple[int, ...]  # expected output shape from correct op
    is_linear: bool
    mismatch_types: List[str]         # ["shape","content","domain","nonlinear","dimensional"]
    operator_source: str              # "dedicated" or "graph"


# fmt: off
MODALITY_SPECS: Dict[str, ModalitySpec] = {

    # =========================================================================
    # MICROSCOPY
    # =========================================================================
    "widefield": ModalitySpec(
        modality="widefield",
        category="microscopy",
        display_name="Widefield Fluorescence Microscopy",
        physics_intro=(
            "Standard widefield epi-fluorescence microscopy where the entire field of view "
            "is illuminated simultaneously and the image is formed by convolution of the "
            "specimen fluorescence distribution with the system point spread function (PSF). "
            "Out-of-focus blur is the primary degradation."
        ),
        forward_equation="y = PSF ** x + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=[],  # This IS the fallback
        operator_source="dedicated",
    ),
    "widefield_lowdose": ModalitySpec(
        modality="widefield_lowdose",
        category="microscopy",
        display_name="Low-Dose Widefield Microscopy",
        physics_intro=(
            "Widefield fluorescence microscopy operated at very low illumination power or "
            "short exposure time to reduce phototoxicity. Images are dominated by shot noise "
            "and read noise. The PSF model is identical to standard widefield but the noise "
            "model differs significantly (heavy Poisson + Gaussian)."
        ),
        forward_equation="y = Poisson(alpha * PSF ** x) / alpha + N(0, sigma^2)",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="dedicated",
    ),
    "confocal_livecell": ModalitySpec(
        modality="confocal_livecell",
        category="microscopy",
        display_name="Confocal Live-Cell Microscopy",
        physics_intro=(
            "Laser scanning confocal microscopy for live-cell imaging. A focused laser "
            "scans the specimen point by point, and a pinhole rejects out-of-focus light. "
            "The confocal PSF is sharper (sigma~1.2-1.5) than widefield (sigma=2.0), "
            "producing fundamentally different blur characteristics."
        ),
        forward_equation="y = PSF_confocal ** x + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="dedicated",
    ),
    "confocal_3d": ModalitySpec(
        modality="confocal_3d",
        category="microscopy",
        display_name="Confocal 3D Z-Stack",
        physics_intro=(
            "Three-dimensional confocal imaging by acquiring a z-stack of optical sections. "
            "Each slice is convolved with the 3D confocal PSF. The anisotropic PSF (worse "
            "axial resolution) and volumetric data are key differences from 2D widefield. "
            "The correct operator works on 3D volumes, not 2D images."
        ),
        forward_equation="y(x,y,z) = PSF_3d *** x(x,y,z) + n",
        correct_x_shape=(32, 64, 64),
        correct_y_shape=(32, 64, 64),
        is_linear=True,
        mismatch_types=["dimensional", "content"],
        operator_source="dedicated",
    ),
    "sim": ModalitySpec(
        modality="sim",
        category="microscopy",
        display_name="Structured Illumination Microscopy",
        physics_intro=(
            "Structured Illumination Microscopy achieves ~2x lateral resolution improvement "
            "by illuminating with sinusoidal patterns at multiple orientations and phases. "
            "The forward model produces multiple patterned images (n_angles * n_phases = 9), "
            "fundamentally different from a single blurred output."
        ),
        forward_equation="y_k = PSF ** (I_k * x) + n, k=1..9",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64, 9),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="dedicated",
    ),
    "lightsheet": ModalitySpec(
        modality="lightsheet",
        category="microscopy",
        display_name="Light-Sheet Fluorescence Microscopy",
        physics_intro=(
            "Light-sheet (SPIM) illuminates the sample with a thin sheet of light "
            "perpendicular to the detection axis, providing intrinsic optical sectioning. "
            "The correct operator works on 3D volumes with anisotropic PSF blur and "
            "stripe artifacts — a 2D Gaussian blur loses all volumetric information."
        ),
        forward_equation="y = S(z) * (PSF_3d *** x) + n",
        correct_x_shape=(64, 64, 32),
        correct_y_shape=(64, 64, 32),
        is_linear=True,
        mismatch_types=["dimensional", "content"],
        operator_source="dedicated",
    ),

    # =========================================================================
    # COMPRESSIVE / SPECTRAL
    # =========================================================================
    "cassi": ModalitySpec(
        modality="cassi",
        category="compressive",
        display_name="Coded Aperture Snapshot Spectral Imaging (CASSI)",
        physics_intro=(
            "CASSI compresses a 3D spectral data cube (H x W x L) into a 2D coded "
            "measurement via a binary coded aperture mask and spectral dispersion. "
            "The output shape differs from the input due to dispersive shift. The "
            "widefield fallback ignores spectral encoding entirely."
        ),
        forward_equation="y(x,y) = sum_l M(x,y) * X(x, y-s(l), l)",
        correct_x_shape=(64, 64, 8),
        correct_y_shape=(64, 71),  # dispersed output
        is_linear=True,
        mismatch_types=["shape", "dimensional", "content"],
        operator_source="dedicated",
    ),
    "spc": ModalitySpec(
        modality="spc",
        category="compressive",
        display_name="Single-Pixel Camera",
        physics_intro=(
            "The single-pixel camera acquires compressive measurements by projecting "
            "the scene onto a sequence of random binary patterns and recording a single "
            "scalar intensity per pattern. The output is a 1D measurement vector, not "
            "a 2D image — fundamentally incompatible with the widefield 2D blur model."
        ),
        forward_equation="y = A @ vec(x)",
        correct_x_shape=(64, 64),
        correct_y_shape=(614,),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="dedicated",
    ),
    "cacti": ModalitySpec(
        modality="cacti",
        category="compressive",
        display_name="CACTI (Video Snapshot Compressive Imaging)",
        physics_intro=(
            "CACTI compresses a video sequence (H x W x T frames) into a single 2D "
            "snapshot via time-varying coded apertures. Each temporal frame is modulated "
            "by a different mask pattern and all frames are summed on the detector. "
            "The widefield fallback has no temporal coding structure."
        ),
        forward_equation="y(x,y) = sum_t M_t(x,y) * X(x,y,t)",
        correct_x_shape=(64, 64, 8),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["dimensional", "content"],
        operator_source="dedicated",
    ),
    "matrix": ModalitySpec(
        modality="matrix",
        category="compressive",
        display_name="Generic Matrix Sensing",
        physics_intro=(
            "Generic matrix operator where a random measurement matrix A maps the "
            "vectorized image to a compressed measurement vector. The output is a 1D "
            "vector of length M << N, losing all spatial structure that the widefield "
            "blur model preserves."
        ),
        forward_equation="y = A @ x + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(1024,),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="dedicated",
    ),

    # =========================================================================
    # TOMOGRAPHY / MRI
    # =========================================================================
    "ct": ModalitySpec(
        modality="ct",
        category="medical",
        display_name="X-ray Computed Tomography",
        physics_intro=(
            "X-ray CT acquires line integrals of the attenuation coefficient at "
            "multiple angles via the Radon transform. The output is a sinogram of "
            "shape (n_angles, n_detectors), fundamentally different from the 2D input. "
            "The widefield blur cannot represent angular projections."
        ),
        forward_equation="y(theta, s) = integral x(r) dl (Radon transform)",
        correct_x_shape=(64, 64),
        correct_y_shape=(180, 64),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="dedicated",
    ),
    "mri": ModalitySpec(
        modality="mri",
        category="medical",
        display_name="Magnetic Resonance Imaging",
        physics_intro=(
            "MRI acquires data in the spatial frequency domain (k-space) via the "
            "Fourier transform, with undersampling via a binary mask. The output is "
            "complex-valued undersampled k-space data — the widefield fallback produces "
            "only real-valued spatially blurred output, losing phase information entirely."
        ),
        forward_equation="y = M * F * x + n (k-space undersampling)",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["domain", "content"],
        operator_source="dedicated",
    ),

    # =========================================================================
    # PHASE / HOLOGRAPHY / COHERENT
    # =========================================================================
    "ptychography": ModalitySpec(
        modality="ptychography",
        category="coherent",
        display_name="Ptychographic Imaging",
        physics_intro=(
            "Ptychography is a scanning coherent imaging technique where overlapping "
            "probe positions illuminate the sample. Each diffraction pattern is the "
            "squared modulus of the Fourier transform of the probe-sample product. "
            "The non-linear magnitude-squared detection makes this fundamentally "
            "different from linear convolution."
        ),
        forward_equation="I_j = |FFT(P(r) * O(r - r_j))|^2",
        correct_x_shape=(64, 64),
        correct_y_shape=(16, 32, 32),
        is_linear=False,
        mismatch_types=["shape", "nonlinear", "content"],
        operator_source="dedicated",
    ),
    "holography": ModalitySpec(
        modality="holography",
        category="coherent",
        display_name="Digital Holographic Microscopy",
        physics_intro=(
            "Digital holographic microscopy records the interference pattern between "
            "the object wave and a reference wave. The complex-valued wave propagation "
            "(Fresnel/angular spectrum) produces an interference hologram. The widefield "
            "fallback loses all phase information by producing only real-valued output."
        ),
        forward_equation="I = |U_obj * exp(i*phi) + U_ref|^2",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["domain", "content"],
        operator_source="dedicated",
    ),
    "phase_retrieval": ModalitySpec(
        modality="phase_retrieval",
        category="coherent",
        display_name="Coherent Diffractive Imaging (CDI)",
        physics_intro=(
            "Phase retrieval / CDI measures only the intensity (squared modulus) of "
            "the Fourier transform of the sample transmittance. This is an inherently "
            "non-linear forward model (loss of phase). The widefield linear convolution "
            "cannot represent Fourier intensity measurement."
        ),
        forward_equation="y = |F{x}|^2",
        correct_x_shape=(64, 64),
        correct_y_shape=(128, 128),
        is_linear=False,
        mismatch_types=["shape", "nonlinear", "content"],
        operator_source="dedicated",
    ),
    "fpm": ModalitySpec(
        modality="fpm",
        category="coherent",
        display_name="Fourier Ptychographic Microscopy",
        physics_intro=(
            "FPM illuminates the sample from multiple angles to synthetically extend "
            "the microscope's numerical aperture. Each angle captures |F^{-1}{P(k-k_j) * O(k)}|^2, "
            "a non-linear intensity measurement. The widefield blur captures none of "
            "the angular diversity or Fourier synthesis."
        ),
        forward_equation="y_j = |F^{-1}{P(k - k_j) * O(k)}|^2",
        correct_x_shape=(64, 64),
        correct_y_shape=(16, 16),  # lr_size = hr_size // 4
        is_linear=False,
        mismatch_types=["shape", "nonlinear", "content"],
        operator_source="dedicated",
    ),

    # =========================================================================
    # RENDERING / 3D
    # =========================================================================
    "nerf": ModalitySpec(
        modality="nerf",
        category="neural_rendering",
        display_name="Neural Radiance Fields (NeRF)",
        physics_intro=(
            "NeRF represents a scene as a continuous volumetric function mapping 3D "
            "coordinates to density and color. Rendering integrates along camera rays "
            "through the volume. The correct operator takes a 3D volume and produces "
            "multiple 2D views — the widefield 2D blur is dimensionally incompatible."
        ),
        forward_equation="C(r) = integral T(t) * sigma(r(t)) * c(r(t),d) dt",
        correct_x_shape=(64, 64, 32),
        correct_y_shape=(10, 64, 64),
        is_linear=False,
        mismatch_types=["dimensional", "nonlinear", "shape", "content"],
        operator_source="dedicated",
    ),
    "gaussian_splatting": ModalitySpec(
        modality="gaussian_splatting",
        category="neural_rendering",
        display_name="3D Gaussian Splatting",
        physics_intro=(
            "3D Gaussian Splatting renders scenes by projecting anisotropic 3D Gaussians "
            "onto the image plane with alpha compositing. Like NeRF, it takes a 3D volume "
            "and renders multi-view 2D images. The non-linear splatting+compositing is "
            "fundamentally different from a 2D Gaussian blur."
        ),
        forward_equation="C(p) = sum_i alpha_i * c_i * prod_{j<i}(1 - alpha_j)",
        correct_x_shape=(64, 64, 32),
        correct_y_shape=(10, 64, 64),
        is_linear=False,
        mismatch_types=["dimensional", "nonlinear", "shape", "content"],
        operator_source="dedicated",
    ),

    # =========================================================================
    # LENSLESS / COMPUTATIONAL
    # =========================================================================
    "lensless": ModalitySpec(
        modality="lensless",
        category="computational",
        display_name="Lensless (Diffuser Camera) Imaging",
        physics_intro=(
            "Lensless imaging replaces the lens with a coded optical element (diffuser "
            "or mask), producing a heavily convolved measurement. The PSF is typically "
            "much larger (sigma~10) and spatially varying compared to widefield (sigma=2.0), "
            "resulting in fundamentally different blur characteristics."
        ),
        forward_equation="y = PSF_diffuser ** x + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="dedicated",
    ),
    "panorama": ModalitySpec(
        modality="panorama",
        category="computational",
        display_name="Panorama Multi-Focus Fusion",
        physics_intro=(
            "Panoramic imaging captures multiple focal planes and stitches them into "
            "a single all-in-focus composite. The forward model includes geometric "
            "warping (parallax shifts) and depth-dependent blur — neither of which "
            "is captured by the simple Gaussian convolution fallback."
        ),
        forward_equation="y_k = PSF(d_k) ** x + n, k=1..N_focus",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "light_field": ModalitySpec(
        modality="light_field",
        category="computational",
        display_name="Light Field Imaging",
        physics_intro=(
            "Light field cameras use a microlens array to capture both spatial and "
            "angular information of the light field. The forward model involves "
            "disparity-dependent shifts and microlens integration that produce "
            "fundamentally different measurements from simple blur."
        ),
        forward_equation="y(x,y) = sum_{u,v} L(x,y,u,v) * MLA(x,y,u,v)",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64, 5, 5),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="dedicated",
    ),

    # =========================================================================
    # BIOMEDICAL
    # =========================================================================
    "dot": ModalitySpec(
        modality="dot",
        category="medical",
        display_name="Diffuse Optical Tomography",
        physics_intro=(
            "DOT reconstructs internal tissue optical properties from surface "
            "boundary measurements. Multiple source-detector pairs on the surface "
            "measure diffuse photon propagation through tissue. The output is a 1D "
            "measurement vector — not a 2D blurred image."
        ),
        forward_equation="y = J(mu_a, mu_s') * x",
        correct_x_shape=(16, 16),
        correct_y_shape=(64,),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="dedicated",
    ),
    "photoacoustic": ModalitySpec(
        modality="photoacoustic",
        category="medical",
        display_name="Photoacoustic Imaging",
        physics_intro=(
            "Photoacoustic imaging converts pulsed laser illumination into "
            "acoustic waves via thermoelastic expansion. Ultrasonic transducers "
            "detect the acoustic signals. The forward model involves a circular "
            "Radon transform producing time-series data for each transducer, "
            "not a spatially blurred 2D image."
        ),
        forward_equation="y = R * p0, p0 = Gamma * mu_a * Phi",
        correct_x_shape=(64, 64),
        correct_y_shape=(32, 128),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="dedicated",
    ),
    "oct": ModalitySpec(
        modality="oct",
        category="medical",
        display_name="Optical Coherence Tomography",
        physics_intro=(
            "OCT uses low-coherence interferometry to obtain depth-resolved images. "
            "The forward model involves spectral modulation and Fourier transform "
            "relations between sample reflectivity and detected spectral interferogram. "
            "The widefield blur cannot represent the spectral encoding."
        ),
        forward_equation="y(k) = |E_r + E_s(k)|^2",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),  # OCT dedicated op reshapes internally
        is_linear=False,
        mismatch_types=["content"],
        operator_source="dedicated",
    ),
    "flim": ModalitySpec(
        modality="flim",
        category="microscopy",
        display_name="Fluorescence Lifetime Imaging",
        physics_intro=(
            "FLIM measures the fluorescence decay kinetics at each pixel, producing "
            "a 3D data cube (spatial x temporal). The forward model convolves the "
            "multi-exponential decay with the instrument response function. The "
            "widefield blur ignores temporal dynamics entirely."
        ),
        forward_equation="y(t) = IRF(t) * [sum_i a_i * exp(-t/tau_i)]",
        correct_x_shape=(32, 32),
        correct_y_shape=(32, 32, 64),
        is_linear=False,
        mismatch_types=["shape", "nonlinear", "content"],
        operator_source="dedicated",
    ),
    "integral": ModalitySpec(
        modality="integral",
        category="computational",
        display_name="Integral Photography (Plenoptic)",
        physics_intro=(
            "Integral photography (plenoptic imaging) uses a microlens array to "
            "capture multi-view sub-images at different depths. The forward model "
            "produces a set of depth-dependent views, which the widefield fallback "
            "collapses into a single blurred 2D image."
        ),
        forward_equation="I(x,y) = integral L(x,y,u,v) * T(u,v) dudv",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64, 8),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="dedicated",
    ),

    # =========================================================================
    # NEW DEDICATED OPERATORS
    # =========================================================================
    "xray_radiography": ModalitySpec(
        modality="xray_radiography",
        category="medical",
        display_name="X-ray Radiography",
        physics_intro=(
            "X-ray radiography measures the transmission of X-rays through tissue "
            "following Beer-Lambert exponential attenuation. The non-linear "
            "exponential model (I = I_0 * exp(-mu*x)) produces contrast fundamentally "
            "different from linear Gaussian convolution."
        ),
        forward_equation="y = eta * I_0 * exp(-mu * x) + scatter + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="dedicated",
    ),
    "ultrasound": ModalitySpec(
        modality="ultrasound",
        category="medical",
        display_name="Ultrasound Imaging",
        physics_intro=(
            "Ultrasound imaging sends acoustic pulses into tissue and records "
            "echo signals at an array of transducer elements. The output is "
            "RF channel data of shape (n_elements, n_samples) — a fundamentally "
            "different domain from a spatially blurred 2D image."
        ),
        forward_equation="y = DAS(H * x) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(32, 128),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="dedicated",
    ),
    "pet": ModalitySpec(
        modality="pet",
        category="medical",
        display_name="Positron Emission Tomography",
        physics_intro=(
            "PET detects annihilation photon pairs to reconstruct radiotracer "
            "distribution. The forward model projects the activity distribution "
            "at multiple angles producing a sinogram. The widefield blur has no "
            "concept of angular projections."
        ),
        forward_equation="y = Poisson(A * x + scatter)",
        correct_x_shape=(64, 64),
        correct_y_shape=(32, 64),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="dedicated",
    ),
    "spect": ModalitySpec(
        modality="spect",
        category="medical",
        display_name="Single Photon Emission CT",
        physics_intro=(
            "SPECT images gamma-ray emitting radiotracers using collimated detectors "
            "that rotate around the patient. The collimator creates a depth-dependent "
            "blur in the projection, producing sinogram data incompatible with "
            "simple 2D Gaussian convolution."
        ),
        forward_equation="y = Poisson(C * A * x)",
        correct_x_shape=(64, 64),
        correct_y_shape=(32, 64),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="dedicated",
    ),
    "sem": ModalitySpec(
        modality="sem",
        category="electron_microscopy",
        display_name="Scanning Electron Microscopy",
        physics_intro=(
            "SEM forms images by scanning a focused electron beam across the sample "
            "and detecting secondary or backscattered electrons. The signal depends "
            "on material properties (yield coefficient) and beam-sample interaction, "
            "not optical PSF convolution."
        ),
        forward_equation="y = G * eta(E_0) * I_b * x + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="dedicated",
    ),
    "tem": ModalitySpec(
        modality="tem",
        category="electron_microscopy",
        display_name="Transmission Electron Microscopy",
        physics_intro=(
            "TEM transmits an electron beam through a thin specimen. Image contrast "
            "arises from the contrast transfer function (CTF) applied in Fourier space, "
            "which produces complex-valued output with oscillating phase contrast. "
            "The widefield real-valued blur cannot represent CTF oscillations."
        ),
        forward_equation="y = |IFFT(CTF * FFT(exp(i*sigma*V*x)))|^2 + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["domain", "content"],
        operator_source="dedicated",
    ),
    "electron_tomography": ModalitySpec(
        modality="electron_tomography",
        category="electron_microscopy",
        display_name="Electron Tomography",
        physics_intro=(
            "Electron tomography acquires a tilt series of TEM projections at "
            "multiple angles through a 3D specimen volume. The forward model is "
            "a 3D-to-2D projection operator — the widefield 2D blur has no concept "
            "of tilt-angle projections through a volume."
        ),
        forward_equation="y_i = R(angle_i) * volume + n_i",
        correct_x_shape=(32, 64, 64),
        correct_y_shape=(16, 64, 64),
        is_linear=True,
        mismatch_types=["dimensional", "shape", "content"],
        operator_source="dedicated",
    ),

    # =========================================================================
    # GRAPH-ONLY V2 MODALITIES
    # =========================================================================
    "stem": ModalitySpec(
        modality="stem",
        category="electron_microscopy",
        display_name="Scanning TEM (STEM)",
        physics_intro=(
            "STEM scans a focused electron probe across the sample, detecting "
            "transmitted electrons with an annular detector. Image contrast depends "
            "on probe convolution and atomic-number-dependent scattering (Z-contrast), "
            "not simple Gaussian blur."
        ),
        forward_equation="y = G * eta * (probe ** x) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "fluoroscopy": ModalitySpec(
        modality="fluoroscopy",
        category="medical",
        display_name="Fluoroscopy",
        physics_intro=(
            "Fluoroscopy provides real-time X-ray imaging via Beer-Lambert "
            "attenuation with temporal integration of multiple low-dose frames. "
            "The exponential attenuation and temporal averaging produce contrast "
            "unlike simple Gaussian convolution."
        ),
        forward_equation="y = (1/N) * sum_t eta * I_0 * exp(-mu * x_t) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "mammography": ModalitySpec(
        modality="mammography",
        category="medical",
        display_name="Mammography",
        physics_intro=(
            "Mammography uses low-energy X-rays optimized for breast tissue imaging. "
            "The Beer-Lambert attenuation model with tissue-specific absorption "
            "coefficients produces fundamentally different contrast from Gaussian blur."
        ),
        forward_equation="y = eta * I_0 * exp(-mu(E) * x) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "dexa": ModalitySpec(
        modality="dexa",
        category="medical",
        display_name="Dual-Energy X-ray Absorptiometry (DEXA)",
        physics_intro=(
            "DEXA acquires X-ray images at two energies to separate bone and soft "
            "tissue contributions. The output is a dual-channel image (2, H, W), "
            "fundamentally different from a single-channel blurred image."
        ),
        forward_equation="y_E = I_0(E) * exp(-(mu_bone*x_bone + mu_soft*x_soft)) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(2, 64, 64),
        is_linear=False,
        mismatch_types=["shape", "content"],
        operator_source="graph",
    ),
    "cbct": ModalitySpec(
        modality="cbct",
        category="medical",
        display_name="Cone-Beam CT",
        physics_intro=(
            "Cone-beam CT uses a divergent X-ray beam to acquire projections at "
            "multiple angles, producing a sinogram. Like fan-beam CT, the output "
            "shape (n_angles, n_detectors) is fundamentally different from the "
            "2D input image."
        ),
        forward_equation="y(theta, s) = integral x(r) dl (cone-beam Radon)",
        correct_x_shape=(64, 64),
        correct_y_shape=(180, 64),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="graph",
    ),
    "angiography": ModalitySpec(
        modality="angiography",
        category="medical",
        display_name="X-ray Angiography",
        physics_intro=(
            "X-ray angiography visualizes blood vessels using contrast agent injection. "
            "The forward model involves Beer-Lambert attenuation with time-dependent "
            "contrast agent concentration and temporal subtraction (DSA), unlike "
            "simple Gaussian blur."
        ),
        forward_equation="y = eta * I_0 * exp(-mu_tissue*x - mu_contrast*c(t)) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "doppler_ultrasound": ModalitySpec(
        modality="doppler_ultrasound",
        category="medical",
        display_name="Doppler Ultrasound",
        physics_intro=(
            "Doppler ultrasound measures blood flow velocity from the frequency "
            "shift of reflected ultrasound pulses. The output shape (n_sensors, "
            "n_samples) is a 2D matrix of acoustic data, not a spatially blurred image."
        ),
        forward_equation="f_d = (2 * v * cos(theta) * f_0) / c",
        correct_x_shape=(64, 64),
        correct_y_shape=(32, 64),
        is_linear=False,
        mismatch_types=["shape", "content"],
        operator_source="graph",
    ),
    "elastography": ModalitySpec(
        modality="elastography",
        category="medical",
        display_name="Shear-Wave Elastography",
        physics_intro=(
            "Elastography maps tissue stiffness by tracking shear wave propagation "
            "via ultrasonic imaging. The output is acoustic wave data at sensor "
            "positions (n_sensors, n_samples), incompatible with 2D blur output."
        ),
        forward_equation="c_s = sqrt(G / rho), E = 3 * rho * c_s^2",
        correct_x_shape=(64, 64),
        correct_y_shape=(32, 64),
        is_linear=False,
        mismatch_types=["shape", "content"],
        operator_source="graph",
    ),
    "fmri": ModalitySpec(
        modality="fmri",
        category="medical",
        display_name="Functional MRI (BOLD)",
        physics_intro=(
            "fMRI detects BOLD contrast by measuring T2*-weighted signal changes "
            "in k-space. Like standard MRI, data is acquired in the Fourier domain "
            "with undersampling, producing complex-valued k-space measurements — "
            "not real-valued spatially blurred images."
        ),
        forward_equation="y = M * F * S * (HRF * x) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "mrs": ModalitySpec(
        modality="mrs",
        category="medical",
        display_name="MR Spectroscopy",
        physics_intro=(
            "MRS measures the frequency spectrum of metabolites in a localized "
            "region. Like MRI, data is acquired in k-space via Fourier encoding. "
            "The widefield spatial blur has no concept of spectral information."
        ),
        forward_equation="S(f) = sum_k a_k * L(f - f_k, T2_k) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "diffusion_mri": ModalitySpec(
        modality="diffusion_mri",
        category="medical",
        display_name="Diffusion MRI (DTI)",
        physics_intro=(
            "Diffusion MRI measures water molecule diffusion directionality via "
            "diffusion-encoding gradients in k-space. Each gradient direction "
            "produces a different k-space weighting — the widefield blur ignores "
            "all diffusion encoding."
        ),
        forward_equation="S(g) = S_0 * exp(-b * g^T * D * g) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "two_photon": ModalitySpec(
        modality="two_photon",
        category="microscopy",
        display_name="Two-Photon Microscopy",
        physics_intro=(
            "Two-photon microscopy achieves optical sectioning through non-linear "
            "two-photon absorption (signal proportional to I^2). The quadratic "
            "dependence on excitation intensity makes the forward model inherently "
            "non-linear — unlike the linear Gaussian convolution fallback."
        ),
        forward_equation="y = PSF ** |x|^2 + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "sted": ModalitySpec(
        modality="sted",
        category="microscopy",
        display_name="STED Microscopy",
        physics_intro=(
            "STED achieves sub-diffraction resolution by depleting fluorescence "
            "with a donut-shaped depletion beam. The effective PSF is sharpened "
            "beyond the diffraction limit — the widefield sigma=2.0 blur dramatically "
            "over-blurs compared to the STED effective PSF."
        ),
        forward_equation="y = PSF_sted ** (x * (1 - eta * donut)) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "palm_storm": ModalitySpec(
        modality="palm_storm",
        category="microscopy",
        display_name="PALM/STORM Single-Molecule Localization",
        physics_intro=(
            "PALM/STORM achieves nanoscale resolution by stochastically activating "
            "sparse subsets of fluorophores and localizing their positions. The "
            "forward model involves sparse emitter blinking statistics — fundamentally "
            "different from continuous Gaussian convolution."
        ),
        forward_equation="y_t = sum_k I_k(t) * PSF(r - r_k) + bg + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "tirf": ModalitySpec(
        modality="tirf",
        category="microscopy",
        display_name="TIRF Microscopy",
        physics_intro=(
            "Total Internal Reflection Fluorescence (TIRF) selectively excites "
            "fluorophores within ~100nm of the coverslip via evanescent wave. "
            "The exponential axial decay creates depth sectioning absent in "
            "widefield — the uniform blur ignores this axial selectivity."
        ),
        forward_equation="y = PSF ** (x * exp(-z/d)) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "polarization": ModalitySpec(
        modality="polarization",
        category="microscopy",
        display_name="Polarization Microscopy",
        physics_intro=(
            "Polarization microscopy measures birefringence and dichroism via "
            "Mueller/Jones matrix formalism. The polarization state transformation "
            "is fundamentally different from scalar Gaussian blur — it encodes "
            "material anisotropy that the scalar fallback cannot represent."
        ),
        forward_equation="y = PSF ** (M * x) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "endoscopy": ModalitySpec(
        modality="endoscopy",
        category="clinical_optics",
        display_name="Fiber Bundle Endoscopy",
        physics_intro=(
            "Endoscopic imaging transmits images through a fiber bundle with "
            "honeycomb sampling and inter-core crosstalk. The output is a 1D "
            "vector of core intensities (n_cores), not a 2D spatially blurred image."
        ),
        forward_equation="y = Poisson(alpha * S(F * x)) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(4096,),
        is_linear=False,
        mismatch_types=["shape", "content"],
        operator_source="graph",
    ),
    "fundus": ModalitySpec(
        modality="fundus",
        category="clinical_optics",
        display_name="Fundus Camera",
        physics_intro=(
            "Fundus photography images the retina through the pupil. While "
            "similar to widefield in form (PSF blur), the retinal PSF accounts "
            "for ocular aberrations and the specific optics of the fundus camera, "
            "differing from a generic sigma=2.0 Gaussian."
        ),
        forward_equation="y = PSF ** x + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "octa": ModalitySpec(
        modality="octa",
        category="clinical_optics",
        display_name="OCT Angiography",
        physics_intro=(
            "OCTA uses repeated OCT scans to detect motion contrast from blood "
            "flow. The forward model involves angular spectrum propagation and "
            "temporal variance estimation — fundamentally different from single-frame "
            "Gaussian blur."
        ),
        forward_equation="y = Var_t[OCT(x, t)] + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "tof_camera": ModalitySpec(
        modality="tof_camera",
        category="depth_imaging",
        display_name="Time-of-Flight Depth Camera",
        physics_intro=(
            "ToF cameras measure scene depth by timing modulated light round-trips. "
            "The forward model involves temporal gating and SPAD detection with "
            "timing jitter — not spatial Gaussian convolution."
        ),
        forward_equation="y = Poisson(alpha * G(x)) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "lidar": ModalitySpec(
        modality="lidar",
        category="depth_imaging",
        display_name="LiDAR Scanner",
        physics_intro=(
            "LiDAR acquires 3D point clouds by scanning a laser beam and timing "
            "return pulses. The output is a 1D range profile per scan line — not "
            "a 2D spatially blurred image."
        ),
        forward_equation="y = Poisson(alpha * T(S(x))) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64,),
        is_linear=False,
        mismatch_types=["shape", "content"],
        operator_source="graph",
    ),
    "structured_light": ModalitySpec(
        modality="structured_light",
        category="depth_imaging",
        display_name="Structured-Light Depth Camera",
        physics_intro=(
            "Structured light projects coded fringe patterns onto the scene and "
            "triangulates depth from pattern deformation. The projective geometry "
            "and fringe analysis are fundamentally different from isotropic "
            "Gaussian blur."
        ),
        forward_equation="y = O(P(x)) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "sar": ModalitySpec(
        modality="sar",
        category="remote_sensing",
        display_name="Synthetic Aperture Radar",
        physics_intro=(
            "SAR synthesizes a large aperture by coherently combining radar echoes "
            "along the flight path. The complex-valued range-azimuth focusing is "
            "fundamentally different from real-valued Gaussian blur. SAR images "
            "contain speckle and phase information absent in optical images."
        ),
        forward_equation="y = A_sar * x + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "sonar": ModalitySpec(
        modality="sonar",
        category="remote_sensing",
        display_name="Sonar Imaging",
        physics_intro=(
            "Sonar forms images by beamforming acoustic echoes from an array of "
            "transducer elements. The output is a 1D range profile per beam — "
            "not a 2D spatially blurred image."
        ),
        forward_equation="y = B * H * x + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64,),
        is_linear=True,
        mismatch_types=["shape", "content"],
        operator_source="graph",
    ),
    "electron_diffraction": ModalitySpec(
        modality="electron_diffraction",
        category="electron_microscopy",
        display_name="4D-STEM Electron Diffraction",
        physics_intro=(
            "4D-STEM records a diffraction pattern at each probe position. The "
            "far-field diffraction pattern is the squared Fourier modulus of the "
            "transmitted wave — a non-linear operation that the linear Gaussian "
            "blur cannot represent."
        ),
        forward_equation="y = |FFT(t * P)|^2 + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "ebsd": ModalitySpec(
        modality="ebsd",
        category="electron_microscopy",
        display_name="Electron Backscatter Diffraction",
        physics_intro=(
            "EBSD measures crystallographic orientation by analyzing Kikuchi "
            "diffraction patterns from backscattered electrons. The reciprocal-space "
            "geometry of the diffraction patterns is unrelated to simple spatial "
            "Gaussian blur."
        ),
        forward_equation="y = R(theta,phi,psi) * I_ref + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "eels": ModalitySpec(
        modality="eels",
        category="electron_microscopy",
        display_name="Electron Energy Loss Spectroscopy",
        physics_intro=(
            "EELS measures the energy distribution of electrons transmitted through "
            "a thin specimen. The forward model convolves the energy loss spectrum "
            "with the zero-loss peak — a spectral convolution distinct from spatial "
            "Gaussian blur."
        ),
        forward_equation="y = ZLP ** S(E) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=True,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "electron_holography": ModalitySpec(
        modality="electron_holography",
        category="electron_microscopy",
        display_name="Electron Holography",
        physics_intro=(
            "Electron holography records the interference between the transmitted "
            "electron wave and a reference wave, encoding phase information. "
            "The complex-valued interference pattern contains electromagnetic "
            "potential information absent in real-valued Gaussian blur."
        ),
        forward_equation="y = |psi_obj + psi_ref * exp(i*q*x)|^2 + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["domain", "content"],
        operator_source="graph",
    ),
    "neutron_tomo": ModalitySpec(
        modality="neutron_tomo",
        category="particle_imaging",
        display_name="Neutron Radiography/Tomography",
        physics_intro=(
            "Neutron tomography uses neutron beams that are attenuated by nuclear "
            "interactions (complementary to X-ray contrast). The Beer-Lambert "
            "attenuation with neutron cross-sections produces different contrast "
            "from optical Gaussian blur."
        ),
        forward_equation="y = I_0 * exp(-sigma * x) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "proton_radiography": ModalitySpec(
        modality="proton_radiography",
        category="particle_imaging",
        display_name="Proton Radiography",
        physics_intro=(
            "Proton radiography measures the attenuation and multiple Coulomb "
            "scattering of proton beams through matter. The scattering kernel "
            "(MCS blur) and exponential attenuation are fundamentally different "
            "from a simple Gaussian PSF."
        ),
        forward_equation="y = K_mcs ** (I_0 * exp(-sigma * x)) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
    "muon_tomo": ModalitySpec(
        modality="muon_tomo",
        category="particle_imaging",
        display_name="Muon Tomography",
        physics_intro=(
            "Muon tomography tracks cosmic-ray muons through objects, using "
            "scattering angle distributions to reconstruct density. The output "
            "is scattering data at detector positions — not a spatially blurred "
            "image."
        ),
        forward_equation="theta_rms = (13.6/p) * sqrt(L/X_0) + n",
        correct_x_shape=(64, 64),
        correct_y_shape=(64, 64),
        is_linear=False,
        mismatch_types=["content"],
        operator_source="graph",
    ),
}
# fmt: on

# Sanity check: all modalities covered
assert set(MODALITY_SPECS.keys()) == set(ALL_MODALITIES), (
    f"Missing specs: {set(ALL_MODALITIES) - set(MODALITY_SPECS.keys())}; "
    f"Extra specs: {set(MODALITY_SPECS.keys()) - set(ALL_MODALITIES)}"
)


# ---------------------------------------------------------------------------
# Operator builders
# ---------------------------------------------------------------------------

def build_correct_operator(modality: str):
    """Build the correct operator for a given modality."""
    from pwm_core.core.physics_factory import _build_operator_by_id, _try_build_graph_operator

    dedicated_id = MODALITY_TO_DEDICATED.get(modality, modality)
    if dedicated_id in DEDICATED_MODALITIES:
        op = _build_operator_by_id(dedicated_id, (64, 64), {}, None)
        if op is not None:
            return op, "dedicated"

    op = _try_build_graph_operator(modality, (64, 64))
    if op is not None:
        return op, "graph"

    # Last resort: try build_operator_by_id with the modality name
    op = _build_operator_by_id(modality, (64, 64), {}, None)
    return op, "fallback"


def build_widefield_fallback():
    """Build the widefield fallback operator (sigma=2.0)."""
    from pwm_core.physics.microscopy.widefield import WidefieldOperator
    return WidefieldOperator(
        operator_id="widefield_fallback",
        theta={"sigma": 2.0, "mode": "reflect"},
        x_shape=(64, 64),
    )


def _get_x_shape(op) -> Tuple[int, ...]:
    """Get operator x_shape."""
    shape = getattr(op, "x_shape", None)
    if shape is not None and shape != (1,):
        return tuple(shape)
    shape = getattr(op, "_x_shape", None)
    if shape is not None and shape != (1,):
        return tuple(shape)
    return (64, 64)


# ---------------------------------------------------------------------------
# Mismatch result data
# ---------------------------------------------------------------------------

@dataclass
class MismatchExample:
    """One mismatch example for a single modality."""
    example_id: int
    title: str
    correct_shape: Tuple[int, ...]
    fallback_shape: Tuple[int, ...]
    shapes_match: bool
    correct_is_complex: bool
    fallback_is_complex: bool
    rmse: Optional[float] = None
    correlation: Optional[float] = None
    max_abs_diff: Optional[float] = None
    relative_rmse: Optional[float] = None
    failure_description: str = ""
    correction_text: str = ""


@dataclass
class ModalityResult:
    """Complete mismatch results for one modality."""
    modality: str
    spec: ModalitySpec
    operator_built: bool = False
    build_error: str = ""
    build_path: str = ""
    actual_x_shape: Tuple[int, ...] = ()
    actual_y_shape: Tuple[int, ...] = ()
    examples: List[MismatchExample] = field(default_factory=list)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Mismatch testing functions
# ---------------------------------------------------------------------------

def compute_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Compute comparison metrics between two arrays (must be same shape)."""
    a_f = np.asarray(a, dtype=np.float64).ravel()
    b_f = np.asarray(b, dtype=np.float64).ravel()

    if a_f.size != b_f.size:
        return {"rmse": None, "correlation": None, "max_abs_diff": None, "relative_rmse": None}

    rmse = float(np.sqrt(np.mean((a_f - b_f) ** 2)))
    norm_a = float(np.sqrt(np.mean(a_f ** 2)))
    relative_rmse = rmse / norm_a if norm_a > 1e-30 else float("inf")
    max_abs_diff = float(np.max(np.abs(a_f - b_f)))

    # Correlation
    a_centered = a_f - a_f.mean()
    b_centered = b_f - b_f.mean()
    denom = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    correlation = float(np.sum(a_centered * b_centered) / denom) if denom > 1e-30 else 0.0

    return {
        "rmse": rmse,
        "correlation": correlation,
        "max_abs_diff": max_abs_diff,
        "relative_rmse": relative_rmse,
    }


def run_example_1_forward_comparison(
    correct_op, fallback_op, spec: ModalitySpec
) -> MismatchExample:
    """Example 1: Forward pass comparison between correct and fallback operator."""
    rng = np.random.default_rng(42)
    x_shape = _get_x_shape(correct_op)
    x = rng.standard_normal(x_shape).astype(np.float32)

    # Correct operator forward
    y_correct = correct_op.forward(x)

    # Fallback operator forward (may need 2D slice for 3D inputs)
    x_2d = x
    if len(x.shape) > 2:
        x_2d = x[..., 0] if x.shape[-1] < x.shape[0] else x[0]
        if len(x_2d.shape) > 2:
            x_2d = x_2d[0]
    if len(x_2d.shape) < 2:
        x_2d = x_2d.reshape(64, 64) if x_2d.size == 64*64 else np.zeros((64, 64), dtype=np.float32)
    y_fallback = fallback_op.forward(x_2d)

    correct_shape = tuple(y_correct.shape)
    fallback_shape = tuple(y_fallback.shape)
    shapes_match = correct_shape == fallback_shape
    correct_complex = np.iscomplexobj(y_correct)
    fallback_complex = np.iscomplexobj(y_fallback)

    # Handle complex for comparison
    y_c = np.abs(y_correct) if correct_complex else np.asarray(y_correct, dtype=np.float64)
    y_f = np.abs(y_fallback) if fallback_complex else np.asarray(y_fallback, dtype=np.float64)

    metrics = {}
    if shapes_match:
        metrics = compute_metrics(y_c, y_f)

    # Build failure description
    failure = ""
    correction = ""
    op_id = getattr(correct_op, "operator_id", spec.modality)

    if not shapes_match:
        failure = (
            f"The correct forward model produces output of shape {correct_shape}, "
            f"but the widefield fallback produces shape {fallback_shape}. "
            f"Any reconstruction algorithm expecting the correct output shape will "
            f"crash or produce meaningless results."
        )
        correction = (
            f"Use the correct `{op_id}` operator which produces the "
            f"physically valid output shape {correct_shape}."
        )
    elif correct_complex and not fallback_complex:
        failure = (
            f"The correct forward model produces complex-valued output (contains "
            f"phase information), but the widefield fallback produces only real-valued "
            f"output. Phase information is critical for this modality."
        )
        correction = (
            f"Use the correct `{op_id}` operator which preserves complex-valued "
            f"output containing both amplitude and phase."
        )
    elif metrics.get("rmse") is not None and metrics["rmse"] > 1e-6:
        failure = (
            f"While output shapes match, the pixel values are fundamentally different. "
            f"RMSE = {metrics['rmse']:.6f}, correlation = {metrics['correlation']:.4f}. "
            f"The widefield blur applies a generic Gaussian PSF that does not capture "
            f"the physics of {spec.display_name}."
        )
        correction = (
            f"Use the correct `{op_id}` operator which implements the physical "
            f"forward model: {spec.forward_equation}."
        )
    else:
        failure = "Minimal mismatch — the widefield fallback closely matches this modality."
        correction = "The widefield fallback is acceptable for this modality."

    return MismatchExample(
        example_id=1,
        title="Forward Pass Comparison",
        correct_shape=correct_shape,
        fallback_shape=fallback_shape,
        shapes_match=shapes_match,
        correct_is_complex=correct_complex,
        fallback_is_complex=fallback_complex,
        rmse=metrics.get("rmse"),
        correlation=metrics.get("correlation"),
        max_abs_diff=metrics.get("max_abs_diff"),
        relative_rmse=metrics.get("relative_rmse"),
        failure_description=failure,
        correction_text=correction,
    )


def run_example_2_roundtrip(
    correct_op, fallback_op, spec: ModalitySpec
) -> MismatchExample:
    """Example 2: Round-trip fidelity (forward -> adjoint) comparison."""
    rng = np.random.default_rng(42)
    x_shape = _get_x_shape(correct_op)
    x = rng.standard_normal(x_shape).astype(np.float32)

    # Correct operator round-trip
    correct_roundtrip_rmse = None
    correct_shape = x_shape
    try:
        y_correct = correct_op.forward(x)
        x_hat_correct = correct_op.adjoint(y_correct)
        x_hat_c = np.real(x_hat_correct) if np.iscomplexobj(x_hat_correct) else x_hat_correct
        correct_shape = tuple(x_hat_c.shape)
        if x_hat_c.shape == x.shape:
            correct_roundtrip_rmse = float(np.sqrt(np.mean(
                (x_hat_c.astype(np.float64) - x.astype(np.float64))**2
            )))
    except (RuntimeError, NotImplementedError):
        correct_roundtrip_rmse = None

    # Fallback operator round-trip
    x_2d = x
    if len(x.shape) > 2:
        x_2d = x[..., 0] if x.shape[-1] < x.shape[0] else x[0]
        if len(x_2d.shape) > 2:
            x_2d = x_2d[0]
    if len(x_2d.shape) < 2:
        x_2d = x_2d.reshape(64, 64) if x_2d.size == 64*64 else np.zeros((64, 64), dtype=np.float32)

    y_fallback = fallback_op.forward(x_2d)
    x_hat_fallback = fallback_op.adjoint(y_fallback)
    fallback_shape = tuple(x_hat_fallback.shape)
    fallback_roundtrip_rmse = float(np.sqrt(np.mean(
        (x_hat_fallback.astype(np.float64) - x_2d.astype(np.float64))**2
    )))

    shapes_match = correct_shape == fallback_shape
    op_id = getattr(correct_op, "operator_id", spec.modality)

    # Build failure description
    failure = ""
    correction = ""

    if correct_roundtrip_rmse is not None and correct_shape != x.shape:
        failure = (
            f"Correct round-trip changes shape from {x.shape} to {correct_shape}. "
            f"The widefield round-trip (blur->blur) simply double-blurs the image at "
            f"shape {fallback_shape}."
        )
        correction = (
            f"Use `{op_id}` forward/adjoint pair which preserves the correct "
            f"angular/spectral/volumetric structure through the round-trip."
        )
    elif correct_roundtrip_rmse is None:
        failure = (
            "The correct operator's adjoint is not available (non-linear model). "
            "The widefield round-trip (blur->blur) simply double-blurs the image, "
            "which does not represent the non-linear inverse problem."
        )
        correction = (
            f"For non-linear modalities like {spec.display_name}, iterative "
            f"algorithms (gradient descent, ADMM) must use the correct forward model."
        )
    elif abs((correct_roundtrip_rmse or 0) - fallback_roundtrip_rmse) < 1e-6:
        failure = "Round-trip fidelity is similar — minimal mismatch."
        correction = "The widefield round-trip is acceptable for this modality."
    else:
        failure = (
            f"Correct round-trip RMSE = {correct_roundtrip_rmse:.6f}, "
            f"fallback round-trip RMSE = {fallback_roundtrip_rmse:.6f}. "
            f"The widefield round-trip loses structure-specific information that "
            f"the correct operator preserves."
        )
        correction = (
            f"Use the paired `{op_id}` forward/adjoint for proper "
            f"forward modeling and reconstruction."
        )

    return MismatchExample(
        example_id=2,
        title="Round-Trip Fidelity",
        correct_shape=correct_shape,
        fallback_shape=fallback_shape,
        shapes_match=shapes_match,
        correct_is_complex=False,
        fallback_is_complex=False,
        rmse=correct_roundtrip_rmse,
        correlation=None,
        max_abs_diff=None,
        relative_rmse=None,
        failure_description=failure,
        correction_text=correction,
    )


def run_example_3_bonus(
    correct_op, fallback_op, spec: ModalitySpec
) -> Optional[MismatchExample]:
    """Example 3: Bonus example for special mismatch types (domain/linearity)."""
    rng = np.random.default_rng(123)
    x_shape = _get_x_shape(correct_op)
    op_id = getattr(correct_op, "operator_id", spec.modality)

    # Domain check: complex vs real
    if "domain" in spec.mismatch_types:
        x = rng.standard_normal(x_shape).astype(np.float32)
        y = correct_op.forward(x)
        if np.iscomplexobj(y):
            imag_energy = float(np.sqrt(np.mean(np.imag(y)**2)))
            real_energy = float(np.sqrt(np.mean(np.real(y)**2)))
            if imag_energy > 1e-10:
                return MismatchExample(
                    example_id=3,
                    title="Domain Mismatch — Complex vs Real Output",
                    correct_shape=tuple(y.shape),
                    fallback_shape=(64, 64),
                    shapes_match=tuple(y.shape) == (64, 64),
                    correct_is_complex=True,
                    fallback_is_complex=False,
                    rmse=None,
                    correlation=None,
                    max_abs_diff=None,
                    relative_rmse=None,
                    failure_description=(
                        f"The correct operator output has significant imaginary component "
                        f"(|imag| RMS = {imag_energy:.6f}, |real| RMS = {real_energy:.6f}). "
                        f"The widefield fallback produces only real-valued output, "
                        f"discarding all phase information."
                    ),
                    correction_text=(
                        f"Use `{op_id}` which correctly handles complex-valued "
                        f"wave propagation and phase contrast."
                    ),
                )

    # Linearity check: non-linear modalities
    if "nonlinear" in spec.mismatch_types:
        x1 = rng.standard_normal(x_shape).astype(np.float32) * 0.5
        x2 = rng.standard_normal(x_shape).astype(np.float32) * 0.5
        alpha, beta = 0.6, 0.4
        try:
            y_combined = correct_op.forward(alpha * x1 + beta * x2)
            y_separate = alpha * correct_op.forward(x1) + beta * correct_op.forward(x2)
            y_c = np.abs(y_combined) if np.iscomplexobj(y_combined) else y_combined.astype(np.float64)
            y_s = np.abs(y_separate) if np.iscomplexobj(y_separate) else y_separate.astype(np.float64)
            if y_c.shape == y_s.shape:
                linearity_err = float(np.sqrt(np.mean((y_c - y_s)**2)))
                if linearity_err > 1e-5:
                    return MismatchExample(
                        example_id=3,
                        title="Linearity Mismatch — Non-linear vs Linear Model",
                        correct_shape=tuple(y_c.shape),
                        fallback_shape=(64, 64),
                        shapes_match=tuple(y_c.shape) == (64, 64),
                        correct_is_complex=np.iscomplexobj(y_combined),
                        fallback_is_complex=False,
                        rmse=linearity_err,
                        correlation=None,
                        max_abs_diff=None,
                        relative_rmse=None,
                        failure_description=(
                            f"The correct operator is non-linear: "
                            f"A(a*x1 + b*x2) != a*A(x1) + b*A(x2). "
                            f"Linearity error = {linearity_err:.6f}. "
                            f"The widefield fallback is always linear (Gaussian blur), "
                            f"missing the non-linear physics entirely."
                        ),
                        correction_text=(
                            f"Use `{op_id}` which correctly models the non-linear "
                            f"forward process: {spec.forward_equation}."
                        ),
                    )
        except Exception:
            pass  # Skip if operator can't handle this test

    return None


# ---------------------------------------------------------------------------
# Main testing loop
# ---------------------------------------------------------------------------

def test_modality(modality: str) -> ModalityResult:
    """Run all mismatch examples for a single modality."""
    spec = MODALITY_SPECS[modality]
    result = ModalityResult(modality=modality, spec=spec)
    t0 = time.time()

    try:
        correct_op, build_path = build_correct_operator(modality)
        result.build_path = build_path
        result.operator_built = True
        result.actual_x_shape = _get_x_shape(correct_op)

        fallback_op = build_widefield_fallback()

        # Example 1: Forward Pass Comparison
        ex1 = run_example_1_forward_comparison(correct_op, fallback_op, spec)
        result.examples.append(ex1)
        result.actual_y_shape = ex1.correct_shape

        # Example 2: Round-Trip Fidelity
        try:
            ex2 = run_example_2_roundtrip(correct_op, fallback_op, spec)
            result.examples.append(ex2)
        except Exception as e:
            result.examples.append(MismatchExample(
                example_id=2, title="Round-Trip Fidelity",
                correct_shape=(), fallback_shape=(64, 64),
                shapes_match=False,
                correct_is_complex=False, fallback_is_complex=False,
                failure_description=f"Round-trip test failed: {e}",
                correction_text="Investigate operator adjoint implementation.",
            ))

        # Example 3: Bonus (conditional)
        try:
            ex3 = run_example_3_bonus(correct_op, fallback_op, spec)
            if ex3 is not None:
                result.examples.append(ex3)
        except Exception:
            pass  # Bonus example is optional

    except Exception as e:
        result.build_error = traceback.format_exc()
        # Still generate metadata-based examples
        result.examples.append(MismatchExample(
            example_id=1, title="Forward Pass Comparison (from metadata)",
            correct_shape=spec.correct_y_shape,
            fallback_shape=(64, 64),
            shapes_match=spec.correct_y_shape == (64, 64),
            correct_is_complex=False, fallback_is_complex=False,
            failure_description=(
                f"Operator build failed: {e}. Based on metadata, correct output shape "
                f"is {spec.correct_y_shape} vs widefield (64, 64)."
            ),
            correction_text=f"Fix the operator build for {modality} and re-run.",
        ))
        result.examples.append(MismatchExample(
            example_id=2, title="Round-Trip Fidelity (from metadata)",
            correct_shape=spec.correct_x_shape,
            fallback_shape=(64, 64),
            shapes_match=len(spec.correct_x_shape) == 2,
            correct_is_complex=False, fallback_is_complex=False,
            failure_description=(
                f"Operator build failed. Correct x_shape is {spec.correct_x_shape} "
                f"vs widefield (64, 64)."
            ),
            correction_text=f"Fix the operator build for {modality} and re-run.",
        ))

    result.duration_ms = (time.time() - t0) * 1000
    return result


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_console_report(results: List[ModalityResult]) -> int:
    """Print summary table to console. Returns count of mismatch modalities."""
    header = f"{'#':<4} {'Modality':<28} {'Build':<8} {'Path':<10} {'Ex1':<12} {'Ex2':<12} {'Ex3':<12} {'Mismatch Types':<30} {'ms':<8}"
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  MODALITY MISMATCH EXAMPLES REPORT")
    print("=" * len(header))
    print(header)
    print(sep)

    n_mismatch = 0
    n_examples_total = 0

    for i, r in enumerate(results):
        build_status = "OK" if r.operator_built else "FAIL"
        ex1_status = "---"
        ex2_status = "---"
        ex3_status = "---"

        for ex in r.examples:
            n_examples_total += 1
            if ex.example_id == 1:
                if not ex.shapes_match:
                    ex1_status = "SHAPE_DIFF"
                elif ex.correct_is_complex and not ex.fallback_is_complex:
                    ex1_status = "DOMAIN_DIFF"
                elif ex.rmse is not None and ex.rmse > 1e-6:
                    ex1_status = f"RMSE={ex.rmse:.4f}"
                else:
                    ex1_status = "MATCH"
            elif ex.example_id == 2:
                if ex.rmse is not None:
                    ex2_status = f"RMSE={ex.rmse:.4f}"
                elif "not available" in ex.failure_description.lower() or "non-linear" in ex.failure_description.lower():
                    ex2_status = "NONLIN"
                else:
                    ex2_status = "N/A"
            elif ex.example_id == 3:
                ex3_status = ex.title[:11]

        mismatch_str = ", ".join(r.spec.mismatch_types) if r.spec.mismatch_types else "none"
        if r.spec.mismatch_types:
            n_mismatch += 1

        print(
            f"{i+1:<4} {r.modality:<28} {build_status:<8} {r.build_path:<10} "
            f"{ex1_status:<12} {ex2_status:<12} {ex3_status:<12} "
            f"{mismatch_str:<30} {r.duration_ms:<8.0f}"
        )

    print(sep)
    print(f"SUMMARY: {len(results)} modalities, {n_examples_total} examples total, "
          f"{n_mismatch} with known mismatches")
    print(f"Builds: {sum(1 for r in results if r.operator_built)} OK, "
          f"{sum(1 for r in results if not r.operator_built)} FAIL")

    # Print build failures
    failures = [r for r in results if not r.operator_built]
    if failures:
        print(f"\n--- BUILD FAILURES ({len(failures)}) ---")
        for r in failures:
            err_last_line = r.build_error.strip().split("\n")[-1] if r.build_error else "unknown"
            print(f"  {r.modality}: {err_last_line}")

    return n_mismatch


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def _fmt_shape(shape: Tuple[int, ...]) -> str:
    return f"({', '.join(str(s) for s in shape)})" if shape else "N/A"


def _fmt_float(val: Optional[float], fmt: str = ".6f") -> str:
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def generate_markdown(results: List[ModalityResult]) -> str:
    """Generate the full modality_mismatch_guide.md content."""
    lines: List[str] = []
    lines.append("# Physics World Model — Modality Mismatch Guide\n")
    lines.append(
        "> Auto-generated by `scripts/test_mismatch_examples.py`. "
        "Do not edit manually.\n"
    )
    lines.append(
        "This document shows, for each of the 64 PWM imaging modalities, what goes "
        "wrong when the **widefield Gaussian blur fallback** (sigma=2.0, shape-preserving, "
        "real-valued, self-adjoint) is used instead of the correct physics operator.\n"
    )

    # -- Overview table --
    lines.append("## Overview\n")
    lines.append("| # | Modality | Category | Build | Mismatch Types | Ex1 Shape Match | Ex1 RMSE | Ex2 RMSE |")
    lines.append("|---|----------|----------|-------|----------------|-----------------|----------|----------|")

    for i, r in enumerate(results):
        build = "OK" if r.operator_built else "FAIL"
        mt = ", ".join(r.spec.mismatch_types) if r.spec.mismatch_types else "none"
        ex1_shape = "---"
        ex1_rmse = "---"
        ex2_rmse = "---"
        for ex in r.examples:
            if ex.example_id == 1:
                ex1_shape = "Yes" if ex.shapes_match else f"**No** {_fmt_shape(ex.correct_shape)} vs {_fmt_shape(ex.fallback_shape)}"
                ex1_rmse = _fmt_float(ex.rmse, ".4f")
            elif ex.example_id == 2:
                ex2_rmse = _fmt_float(ex.rmse, ".4f")
        lines.append(f"| {i+1} | `{r.modality}` | {r.spec.category} | {build} | {mt} | {ex1_shape} | {ex1_rmse} | {ex2_rmse} |")

    lines.append("")

    # -- Mismatch Categories --
    lines.append("## Mismatch Categories\n")
    cat_counts = {"shape": 0, "content": 0, "domain": 0, "nonlinear": 0, "dimensional": 0}
    for r in results:
        for mt in r.spec.mismatch_types:
            if mt in cat_counts:
                cat_counts[mt] += 1

    lines.append("| Type | Description | Count |")
    lines.append("|------|-------------|-------|")
    lines.append(f"| **shape** | Correct output shape differs from (64,64) | {cat_counts['shape']} |")
    lines.append(f"| **domain** | Complex output vs real fallback | {cat_counts['domain']} |")
    lines.append(f"| **content** | Same shape but fundamentally wrong physics values | {cat_counts['content']} |")
    lines.append(f"| **nonlinear** | Linear fallback vs nonlinear correct model | {cat_counts['nonlinear']} |")
    lines.append(f"| **dimensional** | 3D input required vs 2D fallback | {cat_counts['dimensional']} |")
    lines.append("")

    # -- Per-modality sections --
    lines.append("---\n")

    for i, r in enumerate(results):
        spec = r.spec
        lines.append(f"## {i+1}. {spec.display_name} (`{r.modality}`)\n")

        # Physics introduction
        lines.append(f"**Physics**: {spec.physics_intro}\n")
        lines.append(f"**Forward equation**: `{spec.forward_equation}`\n")

        op_name = getattr(r, "build_path", spec.operator_source)
        lines.append(
            f"**Correct operator**: `{r.build_path}` | "
            f"**Input shape**: {_fmt_shape(r.actual_x_shape or spec.correct_x_shape)} | "
            f"**Output shape**: {_fmt_shape(r.actual_y_shape or spec.correct_y_shape)} | "
            f"**Linear**: {'Yes' if spec.is_linear else 'No'}\n"
        )

        if r.build_error:
            lines.append(f"> **Build Error**: `{r.build_error.strip().split(chr(10))[-1]}`\n")

        # Each mismatch example
        for ex in r.examples:
            lines.append(f"### Mismatch Example {ex.example_id}: {ex.title}\n")

            lines.append("| Metric | Correct | Widefield Fallback |")
            lines.append("|--------|---------|-------------------|")
            lines.append(f"| Output shape | {_fmt_shape(ex.correct_shape)} | {_fmt_shape(ex.fallback_shape)} |")
            lines.append(f"| Shapes match | {'Yes' if ex.shapes_match else '**No**'} | — |")

            if ex.example_id == 1:
                lines.append(f"| Complex output | {'Yes' if ex.correct_is_complex else 'No'} | {'Yes' if ex.fallback_is_complex else 'No'} |")
                if ex.rmse is not None:
                    lines.append(f"| RMSE | {_fmt_float(ex.rmse)} | — |")
                    lines.append(f"| Correlation | {_fmt_float(ex.correlation, '.4f')} | — |")
                    lines.append(f"| Max abs diff | {_fmt_float(ex.max_abs_diff)} | — |")
                    lines.append(f"| Relative RMSE | {_fmt_float(ex.relative_rmse, '.4f')} | — |")
            elif ex.example_id == 2:
                if ex.rmse is not None:
                    lines.append(f"| Round-trip RMSE | {_fmt_float(ex.rmse)} | — |")

            lines.append("")
            lines.append(f"**Failure**: {ex.failure_description}\n")
            lines.append(f"**Correction**: {ex.correction_text}\n")

        lines.append("---\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Testing mismatch examples for {len(ALL_MODALITIES)} modalities...")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")

    t_start = time.time()
    results: List[ModalityResult] = []

    for i, modality in enumerate(ALL_MODALITIES):
        print(f"  [{i+1}/{len(ALL_MODALITIES)}] {modality}...", end="", flush=True)
        r = test_modality(modality)
        results.append(r)
        n_ex = len(r.examples)
        status = "OK" if r.operator_built else "FAIL"
        print(f" {status} ({n_ex} examples, {r.duration_ms:.0f}ms)")

    elapsed = time.time() - t_start

    # Console report
    n_mismatch = print_console_report(results)

    # Generate markdown
    md_content = generate_markdown(results)

    # Write docs/modality_mismatch_guide.md
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    os.makedirs(docs_dir, exist_ok=True)
    md_path = os.path.join(docs_dir, "modality_mismatch_guide.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nGenerated: {md_path}")
    print(f"Total examples: {sum(len(r.examples) for r in results)}")
    print(f"Total time: {elapsed:.1f}s")

    # Verify minimum 2 examples per modality
    under_two = [r.modality for r in results if len(r.examples) < 2]
    if under_two:
        print(f"\nWARNING: {len(under_two)} modalities have < 2 examples: {under_two}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
