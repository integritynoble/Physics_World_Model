"""
Modality Router — detect imaging modality from natural language prompts.

Uses weighted keyword matching across all 64 PWM modalities.
No ML dependencies — pure string matching with scoring.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ── Keyword dictionary for all 64 modalities ─────────────────────────────
# Each entry: modality_key -> list of keywords/phrases
# Matching weights: exact key match = 3, phrase match = 1
# Longer phrase matches score higher (length bonus).

_MODALITY_KEYWORDS: dict[str, list[str]] = {
    # ── Compressive ───────────────────────────────────────────────────────
    "cassi": [
        "cassi", "coded aperture snapshot spectral", "snapshot spectral",
        "hyperspectral snapshot", "spectral coding", "coded aperture spectral",
        "dispersive coded aperture", "spectral compressive",
    ],
    "cacti": [
        "cacti", "coded aperture compressive temporal", "video compressive",
        "temporal compressive", "snapshot video", "compressive video",
        "high-speed compressive", "video snapshot",
    ],
    "spc": [
        "spc", "single-pixel camera", "single pixel camera", "single pixel",
        "single-pixel", "bucket detector", "compressive sensing camera",
    ],
    "matrix": [
        "matrix sensing", "generic matrix", "compressive matrix",
        "random sensing matrix", "generic compressive",
    ],
    # ── Medical ───────────────────────────────────────────────────────────
    "ct": [
        "ct", "computed tomography", "ct scan", "x-ray ct", "ct reconstruction",
        "hounsfield", "ct image", "fan beam", "ct recon", "radon transform",
        "sinogram", "filtered back projection", "fbp",
    ],
    "cbct": [
        "cbct", "cone-beam ct", "cone beam ct", "cone-beam computed tomography",
        "dental ct", "cone beam",
    ],
    "mri": [
        "mri", "magnetic resonance imaging", "magnetic resonance", "mr imaging",
        "mri scan", "k-space", "t1-weighted", "t2-weighted", "mri recon",
        "compressed sensing mri", "parallel imaging mri", "sense",
        "brain mri", "knee mri", "cardiac mri",
    ],
    "fmri": [
        "fmri", "functional mri", "functional magnetic resonance", "bold",
        "bold signal", "brain activation", "functional brain",
        "resting state fmri", "task fmri", "resting state",
        "functional mri resting", "bold fmri",
    ],
    "diffusion_mri": [
        "diffusion mri", "dti", "diffusion tensor", "diffusion weighted",
        "dwi", "tractography", "diffusion imaging", "fiber tracking",
        "white matter tracts",
    ],
    "mrs": [
        "mrs", "mr spectroscopy", "magnetic resonance spectroscopy",
        "metabolite spectrum", "spectroscopy mri", "brain metabolite",
        "naa", "creatine peak", "choline peak", "lcmodel",
    ],
    "pet": [
        "pet", "positron emission tomography", "positron emission",
        "pet scan", "pet-ct", "fdg", "pet imaging", "radiotracer",
        "pet recon", "mlem",
    ],
    "spect": [
        "spect", "single photon emission", "spect scan",
        "gamma camera", "spect imaging", "spect recon",
    ],
    "ultrasound": [
        "ultrasound", "ultrasound imaging", "b-mode", "b mode",
        "sonography", "ultrasonography", "ultrasound scan",
        "echo", "echography",
    ],
    "doppler_ultrasound": [
        "doppler ultrasound", "doppler", "color doppler", "power doppler",
        "doppler flow", "blood flow ultrasound", "doppler imaging",
    ],
    "elastography": [
        "elastography", "shear wave", "shear-wave", "tissue stiffness",
        "elastic imaging", "elasticity imaging", "liver stiffness",
    ],
    "fluoroscopy": [
        "fluoroscopy", "fluoroscopic", "real-time x-ray", "real time xray",
        "live x-ray", "c-arm", "fluoroscopic imaging",
    ],
    "angiography": [
        "angiography", "angiogram", "dsa", "digital subtraction",
        "vascular imaging", "vessel imaging", "angiographic",
        "coronary angiography", "cerebral angiography",
    ],
    "xray_radiography": [
        "x-ray", "xray", "radiograph", "radiography", "chest x-ray",
        "chest xray", "x-ray image", "plain film", "projection radiography",
    ],
    "mammography": [
        "mammography", "mammogram", "breast imaging", "breast x-ray",
        "breast screening", "mammographic",
    ],
    "dexa": [
        "dexa", "dxa", "dual-energy x-ray", "dual energy x-ray",
        "bone density", "bone densitometry", "absorptiometry",
    ],
    "dot": [
        "dot", "diffuse optical tomography", "diffuse optical",
        "nir tomography", "near-infrared tomography", "optical tomography",
        "diffuse imaging",
    ],
    "photoacoustic": [
        "photoacoustic", "photoacoustic imaging", "optoacoustic",
        "photoacoustic tomography", "pa imaging", "laser ultrasound",
    ],
    # ── Coherent ──────────────────────────────────────────────────────────
    "ptychography": [
        "ptychography", "ptychographic", "ptycho", "ptychographic imaging",
        "scanning diffraction", "epie", "pie algorithm",
    ],
    "holography": [
        "holography", "holographic", "digital holography", "hologram",
        "off-axis holography", "holographic microscopy", "dhm",
        "angular spectrum method",
    ],
    "phase_retrieval": [
        "phase retrieval", "cdi", "coherent diffractive imaging",
        "coherent diffraction", "gerchberg-saxton", "hio",
        "hybrid input-output", "lensless coherent",
    ],
    # ── Microscopy ────────────────────────────────────────────────────────
    "widefield": [
        "widefield", "wide-field", "widefield fluorescence",
        "widefield microscopy", "epifluorescence", "fluorescence microscopy",
        "fluorescence deconvolution",
    ],
    "widefield_lowdose": [
        "low-dose widefield", "low dose widefield", "low-dose microscopy",
        "low dose microscopy", "low dose fluorescence", "photon-limited microscopy",
        "low snr microscopy", "low-dose", "low dose", "low-dose widefield microscopy",
    ],
    "confocal_livecell": [
        "confocal live", "confocal live-cell", "confocal live cell",
        "live-cell confocal", "live cell imaging", "time-lapse confocal",
    ],
    "confocal_3d": [
        "confocal 3d", "confocal z-stack", "confocal z stack",
        "confocal volume", "3d confocal", "confocal",
    ],
    "sim": [
        "sim", "structured illumination", "structured illumination microscopy",
        "sr-sim", "2d-sim", "3d-sim",
    ],
    "lightsheet": [
        "lightsheet", "light-sheet", "light sheet", "spim",
        "lightsheet fluorescence", "selective plane illumination",
        "light sheet microscopy", "light sheet fluorescence microscopy",
        "light-sheet fluorescence microscopy",
    ],
    "two_photon": [
        "two-photon", "two photon", "multiphoton", "multi-photon",
        "2-photon", "2p microscopy", "two-photon microscopy",
        "deep tissue imaging",
    ],
    "sted": [
        "sted", "stimulated emission depletion", "sted microscopy",
        "sted nanoscopy", "super-resolution sted",
    ],
    "tirf": [
        "tirf", "total internal reflection", "total internal reflection fluorescence",
        "evanescent wave", "tirf microscopy",
    ],
    "flim": [
        "flim", "fluorescence lifetime", "lifetime imaging",
        "fluorescence lifetime imaging", "phasor flim",
        "time-resolved fluorescence",
    ],
    "fpm": [
        "fpm", "fourier ptychographic", "fourier ptychography",
        "fourier ptychographic microscopy", "synthetic aperture microscopy",
    ],
    "palm_storm": [
        "palm", "storm", "palm/storm", "single-molecule localization",
        "single molecule localization", "smlm", "super-resolution localization",
        "photoactivated localization", "stochastic optical",
    ],
    "polarization": [
        "polarization microscopy", "polarimetric", "polarization imaging",
        "birefringence", "polarization", "mueller matrix",
    ],
    "lensless": [
        "lensless", "lensless camera", "diffuser camera", "lensless imaging",
        "mask-based imaging", "flatcam",
    ],
    # ── Electron Microscopy ───────────────────────────────────────────────
    "sem": [
        "sem", "scanning electron microscopy", "scanning electron",
        "sem image", "secondary electron",
    ],
    "tem": [
        "tem", "transmission electron microscopy", "transmission electron",
        "tem image", "high-resolution tem", "hrtem",
    ],
    "stem": [
        "stem", "scanning transmission electron", "scanning tem",
        "haadf", "stem image", "annular dark field",
    ],
    "electron_tomography": [
        "electron tomography", "electron tilt series", "et reconstruction",
        "cryo-et", "cryo electron tomography",
    ],
    "electron_diffraction": [
        "electron diffraction", "4d-stem", "4d stem",
        "convergent beam electron diffraction", "cbed",
        "electron diffraction pattern",
    ],
    "electron_holography": [
        "electron holography", "electron holographic", "electron hologram",
        "off-axis electron holography", "electron biprism",
        "lorentz microscopy", "electron phase imaging",
    ],
    "ebsd": [
        "ebsd", "electron backscatter diffraction", "backscatter diffraction",
        "grain orientation", "orientation mapping", "crystallographic mapping",
    ],
    "eels": [
        "eels", "electron energy loss", "energy loss spectroscopy",
        "eels spectrum", "core-loss", "low-loss eels",
    ],
    # ── Clinical Optics ───────────────────────────────────────────────────
    "oct": [
        "oct", "optical coherence tomography", "oct scan",
        "retinal oct", "oct b-scan", "spectral domain oct", "sd-oct",
        "swept source oct",
    ],
    "octa": [
        "octa", "oct angiography", "oct-a", "retinal angiography oct",
        "octa scan", "retinal vasculature oct",
    ],
    "fundus": [
        "fundus", "fundus camera", "fundus image", "retinal image",
        "fundus photography", "optic disc", "retinal photograph",
    ],
    "endoscopy": [
        "endoscopy", "endoscope", "fiber bundle", "endoscopic imaging",
        "fiber endoscope", "capsule endoscopy",
    ],
    # ── Computational / Computational Photography ─────────────────────────
    "light_field": [
        "light field", "light-field", "plenoptic", "light field camera",
        "lenslet array", "plenoptic camera",
    ],
    "integral": [
        "integral photography", "integral imaging", "fly-eye lens",
        "micro-lens array imaging",
    ],
    "panorama": [
        "panorama", "panoramic", "multi-focus fusion", "focus stacking",
        "all-in-focus", "extended depth of field",
    ],
    # ── Neural Rendering ──────────────────────────────────────────────────
    "nerf": [
        "nerf", "neural radiance field", "neural radiance fields",
        "neural radiance", "view synthesis nerf", "novel view synthesis",
        "neural volume rendering",
    ],
    "gaussian_splatting": [
        "gaussian splatting", "3d gaussian", "3dgs", "gaussian splat",
        "3d gaussian splatting", "point-based rendering",
    ],
    # ── Depth Imaging ─────────────────────────────────────────────────────
    "tof_camera": [
        "tof", "time-of-flight", "time of flight", "tof camera",
        "tof depth", "time-of-flight camera", "depth camera tof",
    ],
    "structured_light": [
        "structured light", "structured-light", "fringe projection",
        "structured light depth", "projected pattern depth",
        "phase shifting profilometry",
    ],
    "lidar": [
        "lidar", "laser scanning", "point cloud", "lidar scan",
        "3d scanning", "lidar depth", "aerial lidar",
    ],
    # ── Remote Sensing ────────────────────────────────────────────────────
    "sar": [
        "sar", "synthetic aperture radar", "radar imaging", "sar image",
        "insar", "radar remote sensing", "microwave imaging",
    ],
    "sonar": [
        "sonar", "sonar imaging", "underwater imaging", "acoustic imaging",
        "side-scan sonar", "multibeam sonar", "underwater acoustic",
    ],
    # ── Particle Imaging ──────────────────────────────────────────────────
    "neutron_tomo": [
        "neutron", "neutron tomography", "neutron radiography",
        "neutron imaging", "neutron ct",
    ],
    "proton_radiography": [
        "proton radiography", "proton imaging", "proton ct",
        "proton beam imaging",
    ],
    "muon_tomo": [
        "muon", "muon tomography", "muon imaging", "muon scattering",
        "cosmic ray imaging", "muography",
    ],
}

# Precompute: sort keywords longest-first so longer phrases match before substrings
_KEYWORDS_SORTED: dict[str, list[str]] = {
    mod: sorted(kws, key=len, reverse=True)
    for mod, kws in _MODALITY_KEYWORDS.items()
}


def detect_modality(prompt: str) -> tuple[str, float]:
    """Detect the imaging modality from a natural language prompt.

    Returns (modality_key, confidence) where confidence is 0.0–1.0.
    Returns ("cassi", 0.0) as fallback if nothing matches.
    """
    if not prompt or not prompt.strip():
        return "cassi", 0.0

    text = prompt.lower().strip()
    # Normalize punctuation for matching
    text_clean = re.sub(r"[^\w\s/-]", " ", text)
    text_clean = re.sub(r"\s+", " ", text_clean)

    scores: dict[str, float] = {}

    for mod, keywords in _KEYWORDS_SORTED.items():
        score = 0.0

        # Check exact modality key in text (strong signal)
        # Use word boundary to avoid false positives (e.g., "stem" in "system")
        if re.search(rf"\b{re.escape(mod)}\b", text_clean):
            score += 3.0

        # Check each keyword/phrase
        for kw in keywords:
            if kw in text_clean:
                # Longer phrases are more specific → higher weight
                length_bonus = min(len(kw.split()) * 0.5, 2.0)
                score += 1.0 + length_bonus

        if score > 0:
            scores[mod] = score

    if not scores:
        return "cassi", 0.0

    # ── Disambiguation: prefer specific modalities over generic parents ──
    # If "widefield_lowdose" and "widefield" both score, the specific one
    # should win even if the generic one accumulated more keyword hits.
    # Also handles: holography vs electron_holography, mri vs fmri/diffusion_mri, etc.
    _SPECIFIC_OVERRIDES = {
        # generic -> list of more-specific modalities
        "widefield": ["widefield_lowdose"],
        "holography": ["electron_holography"],
        "mri": ["fmri", "diffusion_mri"],
        "ultrasound": ["doppler_ultrasound", "elastography"],
        "ct": ["cbct"],
        "lightsheet": [],
    }
    for generic, specifics in _SPECIFIC_OVERRIDES.items():
        if generic not in scores:
            continue
        for specific in specifics:
            if specific in scores and scores[specific] >= 1.5:
                # Specific modality matched — suppress the generic one
                scores[generic] = min(scores[generic] * 0.3, scores[specific] - 0.1)

    # Also handle substring-based disambiguation for cases not in the override map
    scored_keys = list(scores.keys())
    for i, mod_a in enumerate(scored_keys):
        for mod_b in scored_keys[i + 1:]:
            if mod_a in mod_b and scores[mod_b] >= 2.0:
                scores[mod_a] = min(scores[mod_a] * 0.3, scores[mod_b] - 0.1)
            elif mod_b in mod_a and scores[mod_a] >= 2.0:
                scores[mod_b] = min(scores[mod_b] * 0.3, scores[mod_a] - 0.1)

    # Pick the highest scoring modality
    best_mod = max(scores, key=scores.get)
    best_score = scores[best_mod]

    # Confidence: normalize by a reasonable max (exact key + 2 keyword matches ≈ 7)
    confidence = min(best_score / 7.0, 1.0)

    logger.info(
        "Modality detection: '%s' → %s (score=%.1f, confidence=%.2f) | top-3: %s",
        prompt[:80],
        best_mod,
        best_score,
        confidence,
        sorted(scores.items(), key=lambda x: -x[1])[:3],
    )

    return best_mod, confidence
