#!/usr/bin/env python3
"""Generate learning material files for all PWM imaging modalities.

Reads each modality's YAML config from benchmarks/configs/ and the
modalities.yaml registry, then creates a 6-file learning folder under
benchmarks/learn/<modality_id>/ with the same structure as the MRI
learning materials.

Usage:
    python benchmarks/learn/generate_all_learn.py
    python benchmarks/learn/generate_all_learn.py --modality ct
    python benchmarks/learn/generate_all_learn.py --skip-existing
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]  # Physics_World_Model
CONFIGS_DIR = _ROOT / "benchmarks" / "configs"
LEARN_DIR = _ROOT / "benchmarks" / "learn"
MODALITIES_YAML = _ROOT / "packages" / "pwm_core" / "contrib" / "modalities.yaml"
SOLVER_REGISTRY = _ROOT / "packages" / "pwm_core" / "contrib" / "solver_registry.yaml"

# ---------------------------------------------------------------------------
# Physics primers by carrier type
# ---------------------------------------------------------------------------
CARRIER_PHYSICS = {
    "X-ray": {
        "intro": (
            "X-rays are high-energy electromagnetic radiation (photon energies "
            "~20-150 keV) that penetrate matter and are attenuated according to "
            "Beer-Lambert law: I = I₀ exp(-∫μ(x,y,z) dl). Different tissues have "
            "different linear attenuation coefficients μ, creating contrast."
        ),
        "key_concepts": [
            "Beer-Lambert attenuation law",
            "Linear attenuation coefficient μ (energy-dependent)",
            "Polychromatic spectrum and beam hardening",
            "Detector types: scintillator + photodiode, photon-counting",
            "Dose considerations: ALARA principle",
        ],
        "signal_equation": "I(d) = I₀ · exp(-∫ μ(l) dl) + noise",
    },
    "Spin/RF": {
        "intro": (
            "Spin/RF imaging exploits nuclear magnetic resonance (NMR). Protons in "
            "a strong B₀ field precess at the Larmor frequency ω = γB₀. RF pulses "
            "tip the magnetisation into the transverse plane, and gradient fields "
            "provide spatial encoding via the Fourier relationship between k-space "
            "and image space."
        ),
        "key_concepts": [
            "Larmor equation: ω = γ B₀ (γ/2π = 42.577 MHz/T for ¹H)",
            "T1 relaxation (spin-lattice), T2 relaxation (spin-spin)",
            "Gradient encoding: frequency, phase, slice selection",
            "k-space: 2D/3D Fourier relationship",
            "Multi-coil arrays and parallel imaging",
        ],
        "signal_equation": "s(t) = ∫∫ ρ(x,y) · S_c(x,y) · e^(-i2π k·r) dr",
    },
    "Photon": {
        "intro": (
            "Photon-based imaging uses visible, near-infrared, or ultraviolet light. "
            "The image formation is typically modelled as convolution with a point "
            "spread function (PSF) determined by the optical system's numerical "
            "aperture and wavelength. Key degradations include diffraction blur, "
            "aberrations, and photon shot noise (Poisson statistics)."
        ),
        "key_concepts": [
            "Diffraction limit: d = 0.61 λ / NA",
            "Point spread function (PSF) and optical transfer function (OTF)",
            "Numerical aperture (NA) and resolution",
            "Shot noise (Poisson) and read noise (Gaussian)",
            "Fluorescence: excitation/emission Stokes shift",
        ],
        "signal_equation": "y = PSF ⊛ x + noise  (⊛ = convolution)",
    },
    "Acoustic": {
        "intro": (
            "Acoustic imaging uses sound waves (1-50 MHz for medical ultrasound, "
            "kHz for sonar). Waves are transmitted into the medium, and reflections "
            "from impedance boundaries are received and beamformed to create images. "
            "The speed of sound (~1540 m/s in tissue) and acoustic impedance "
            "Z = ρ·c determine contrast."
        ),
        "key_concepts": [
            "Acoustic impedance Z = ρ·c and reflection coefficient",
            "Beamforming: delay-and-sum (DAS), adaptive methods",
            "Frequency-dependent attenuation",
            "Phased arrays and synthetic aperture focusing",
            "Doppler effect for flow measurement",
        ],
        "signal_equation": "y(t) = Σ_i  A_i · s(t - 2r_i/c) + noise",
    },
    "Electron": {
        "intro": (
            "Electron imaging uses accelerated electron beams (1-300 keV) whose "
            "de Broglie wavelength (λ = h/√(2meV)) is orders of magnitude shorter "
            "than visible light, enabling atomic-resolution imaging. Electrons "
            "interact with matter via elastic and inelastic scattering, and the "
            "contrast transfer function (CTF) describes phase and amplitude "
            "modulation."
        ),
        "key_concepts": [
            "de Broglie wavelength: λ = h / √(2m_e eV)",
            "Contrast transfer function (CTF)",
            "Elastic vs inelastic scattering",
            "Aberrations: spherical (Cs), chromatic (Cc)",
            "Electron dose and radiation damage",
        ],
        "signal_equation": "I(r) = |F⁻¹{CTF(q) · F{V(r)}}|² + noise",
    },
    "Gamma": {
        "intro": (
            "Gamma-ray imaging detects high-energy photons (140 keV for ⁹⁹ᵐTc, "
            "511 keV for PET) emitted by radioactive tracers inside the body. "
            "Collimation (SPECT) or coincidence detection (PET) provides directional "
            "information. The forward model is based on line integrals of the "
            "tracer distribution, similar to CT but in emission mode."
        ),
        "key_concepts": [
            "Radioactive decay and tracer kinetics",
            "Collimation and coincidence detection",
            "Attenuation correction",
            "Scatter and randoms correction",
            "Resolution recovery and PSF modelling",
        ],
        "signal_equation": "y_i = ∫_Li  f(x) · a(x) dl + scatter + randoms",
    },
    "RF": {
        "intro": (
            "Radiofrequency (RF) imaging uses electromagnetic waves in the MHz-GHz "
            "range. Synthetic aperture radar (SAR) achieves high spatial resolution "
            "by synthesising a large antenna aperture from platform motion. The "
            "signal is coherent, enabling phase-based measurements like "
            "interferometry."
        ),
        "key_concepts": [
            "Range resolution: Δr = c / (2B)",
            "Azimuth resolution and synthetic aperture",
            "Range-Doppler processing",
            "Phase coherence and interferometry",
            "Speckle noise (multiplicative)",
        ],
        "signal_equation": "s(t) = Σ_n  σ_n · exp(-j4πf_c R_n(t)/c) · rect(t/T)",
    },
    "Mechanical": {
        "intro": (
            "Mechanical probe imaging uses physical contact between a sharp tip "
            "and the sample surface. Forces (van der Waals, electrostatic, "
            "magnetic) are measured as the tip scans across the surface. The "
            "tip-sample interaction provides topographic and material property "
            "information with nanometre resolution."
        ),
        "key_concepts": [
            "Tip-sample interaction forces",
            "Contact, tapping, and non-contact modes",
            "Cantilever dynamics and resonance",
            "Piezoelectric scanning and feedback",
            "Tip convolution artefact",
        ],
        "signal_equation": "z(x,y) = h(x,y) ⊛ tip(x,y) + noise",
    },
    "Magnetic": {
        "intro": (
            "Magnetic imaging detects local magnetic fields or magnetic "
            "properties of materials. Techniques range from magnetic force "
            "microscopy (MFM) to magnetic particle imaging (MPI). The signal "
            "depends on the spatial distribution of magnetic moments."
        ),
        "key_concepts": [
            "Magnetic dipole fields and stray fields",
            "Lift-mode scanning (MFM)",
            "Langevin magnetisation curve (MPI)",
            "System function and calibration",
            "Spatial encoding via drive fields",
        ],
        "signal_equation": "u(t) = -dΦ/dt = -μ₀ · ∫ ∂M/∂H · dH/dt · S(r) dr",
    },
    "Ion": {
        "intro": (
            "Ion-based imaging uses focused ion beams or mass spectrometry to "
            "map elemental or molecular composition. Ions are extracted from "
            "the sample surface and analysed by mass-to-charge ratio, providing "
            "spatial chemical maps."
        ),
        "key_concepts": [
            "Sputtering and ion yield",
            "Mass-to-charge ratio analysis",
            "Spatial resolution vs sensitivity trade-off",
            "Matrix effects and quantification",
            "Time-of-flight (ToF) mass analysis",
        ],
        "signal_equation": "I(m/z, x, y) = Y(m/z) · c(x,y) · primary_dose + noise",
    },
    "IR": {
        "intro": (
            "Infrared imaging detects thermal radiation or IR absorption. "
            "Every object above absolute zero emits thermal radiation according "
            "to Planck's law. Active IR techniques use external illumination to "
            "probe material properties via absorption spectroscopy."
        ),
        "key_concepts": [
            "Planck's law and blackbody radiation",
            "Emissivity and thermal contrast",
            "Atmospheric transmission windows",
            "Microbolometer and cooled detector arrays",
            "Thermal diffusivity and heat equation",
        ],
        "signal_equation": "L(λ,T) = (2hc²/λ⁵) · 1/(exp(hc/λkT) - 1)",
    },
    "EM": {
        "intro": (
            "Electromagnetic induction imaging uses alternating magnetic fields "
            "to induce eddy currents in conductive materials. Changes in "
            "impedance due to defects, cracks, or material variations are "
            "detected by the probe coil."
        ),
        "key_concepts": [
            "Eddy current induction and skin depth",
            "Impedance plane analysis",
            "Probe design and lift-off effects",
            "Frequency selection and penetration depth",
            "Defect detection sensitivity",
        ],
        "signal_equation": "ΔZ = ΔR + jΔX = f(σ, μ, geometry, frequency)",
    },
    "THz": {
        "intro": (
            "Terahertz (THz) imaging uses electromagnetic waves at 0.1-10 THz, "
            "bridging the gap between microwave and infrared. THz waves penetrate "
            "many non-metallic materials (plastics, ceramics, textiles) while "
            "being strongly absorbed by water, providing unique contrast."
        ),
        "key_concepts": [
            "THz generation: photoconductive, optical rectification",
            "Time-domain spectroscopy (THz-TDS)",
            "Material-specific absorption signatures",
            "Penetration depth and water sensitivity",
            "Sub-wavelength resolution techniques",
        ],
        "signal_equation": "E(t) = E₀ · h(t) ⊛ sample_response(t) + noise",
    },
    "MV": {
        "intro": (
            "Megavoltage (MV) imaging uses the treatment beam of a linear "
            "accelerator (typically 6 MV) to create portal images for "
            "verification of radiation therapy positioning."
        ),
        "key_concepts": [
            "MV beam characteristics and Compton scattering dominance",
            "Electronic portal imaging devices (EPID)",
            "Low contrast due to Compton dominance",
            "Patient positioning verification",
            "Dose calculation from portal images",
        ],
        "signal_equation": "I = I₀ · exp(-∫ μ_compton(E, ρ) dl) + scatter",
    },
    "Proton": {
        "intro": (
            "Proton imaging uses proton beams (typically 100-250 MeV) that "
            "lose energy as they traverse matter via the Bethe-Bloch formula. "
            "Proton CT measures the residual energy or range to reconstruct "
            "the stopping power distribution."
        ),
        "key_concepts": [
            "Bethe-Bloch energy loss formula",
            "Bragg peak and range",
            "Multiple Coulomb scattering",
            "Stopping power and water-equivalent path length",
            "Comparison with X-ray CT for treatment planning",
        ],
        "signal_equation": "-dE/dx = (4π e⁴ z² N_A Z ρ) / (m_e v²A) · ln(2m_e v²/I)",
    },
    "Seismic/Acoustic": {
        "intro": (
            "Seismic imaging uses low-frequency acoustic/elastic waves "
            "(1-100 Hz) to probe Earth's subsurface structure. Sources generate "
            "compressional (P) and shear (S) waves that reflect and refract at "
            "geological boundaries."
        ),
        "key_concepts": [
            "P-waves and S-waves",
            "Reflection and refraction at interfaces",
            "Normal moveout (NMO) and stacking",
            "Migration: converting time to depth",
            "Full waveform inversion (FWI)",
        ],
        "signal_equation": "∇²u - (1/c²) ∂²u/∂t² = f(x,t)",
    },
    "Seismic": {
        "intro": (
            "Seismic imaging uses low-frequency elastic waves to probe "
            "Earth's interior. Travel times of reflected/refracted waves are "
            "inverted to reconstruct velocity and density structure."
        ),
        "key_concepts": [
            "Seismic wave propagation",
            "Travel time tomography",
            "Ray theory and Fermat's principle",
            "Velocity model building",
            "Resolution kernels",
        ],
        "signal_equation": "t = ∫_ray 1/v(x) dl",
    },
    "Gravitational": {
        "intro": (
            "Gravitational wave detection measures spacetime strain caused by "
            "accelerating masses (merging black holes, neutron stars). Laser "
            "interferometers (LIGO, Virgo) detect differential arm length "
            "changes of ~10⁻²¹ m."
        ),
        "key_concepts": [
            "General relativity and gravitational radiation",
            "Laser interferometry and Michelson configuration",
            "Strain sensitivity: h ~ ΔL/L ~ 10⁻²¹",
            "Matched filtering for signal extraction",
            "Noise: seismic, thermal, shot, quantum",
        ],
        "signal_equation": "h(t) = (ΔL₊ - ΔL₋) / L",
    },
    "Particle": {
        "intro": (
            "Particle calorimetry measures the energy of particles (electrons, "
            "photons, hadrons) by total absorption. Electromagnetic and hadronic "
            "showers develop in dense materials, and the deposited energy is "
            "proportional to the incident particle energy."
        ),
        "key_concepts": [
            "Electromagnetic and hadronic showers",
            "Radiation length and interaction length",
            "Energy resolution: σ_E/E = a/√E ⊕ b ⊕ c/E",
            "Sampling vs homogeneous calorimeters",
            "Particle identification via shower shape",
        ],
        "signal_equation": "E_measured = Σ_cells  w_i · E_cell_i + noise",
    },
    "Electric": {
        "intro": (
            "Electrical impedance imaging applies small alternating currents "
            "through electrodes and measures the resulting voltages to "
            "reconstruct the conductivity distribution inside the body or object."
        ),
        "key_concepts": [
            "Ohm's law: V = I · Z",
            "Conductivity σ and permittivity ε",
            "Ill-posedness and regularisation",
            "Electrode models: complete, shunt, gap",
            "Temporal difference imaging",
        ],
        "signal_equation": "∇·(σ∇φ) = 0  with boundary conditions",
    },
    "Photon/EUV": {
        "intro": (
            "Extreme ultraviolet (EUV) and soft X-ray imaging captures "
            "radiation from hot plasmas (10⁶-10⁷ K). Solar EUV imaging "
            "reveals the Sun's corona, while EUV lithography uses 13.5 nm "
            "light for semiconductor patterning."
        ),
        "key_concepts": [
            "EUV emission from hot plasmas",
            "Multilayer mirror optics",
            "Differential emission measure (DEM)",
            "Coronal temperature diagnostics",
            "EUV absorption in the atmosphere",
        ],
        "signal_equation": "I(λ) = ∫ G(T,λ) · DEM(T) dT",
    },
    "Electron/Photon": {
        "intro": (
            "Coherent diffractive imaging uses coherent beams (electrons or "
            "photons) and records the far-field diffraction pattern. Since "
            "detectors measure only intensity (not phase), phase retrieval "
            "algorithms are needed to reconstruct the complex object."
        ),
        "key_concepts": [
            "Coherent illumination and Fraunhofer diffraction",
            "Phase problem: detectors measure |F{ψ}|² only",
            "Oversampling requirement (>2× Nyquist)",
            "Iterative phase retrieval: HIO, ER, RAAR",
            "Support constraint and positivity",
        ],
        "signal_equation": "I(q) = |F{ψ(r)}|²  (phase lost)",
    },
    "Photon/IR": {
        "intro": (
            "Time-of-flight cameras measure the round-trip time of modulated "
            "light (IR LEDs or lasers) to compute depth. The phase shift of "
            "the reflected signal is proportional to distance."
        ),
        "key_concepts": [
            "Amplitude-modulated continuous wave (AMCW)",
            "Phase-to-depth conversion: d = c·φ/(4πf_mod)",
            "Multi-frequency disambiguation",
            "Systematic errors: multipath, motion blur",
            "Depth resolution and range trade-off",
        ],
        "signal_equation": "d = c · Δφ / (4π f_mod)",
    },
    "Muon": {
        "intro": (
            "Muon tomography uses cosmic-ray muons that scatter as they pass "
            "through dense materials. By tracking incoming and outgoing muon "
            "trajectories, the scattering density (related to atomic number Z) "
            "can be reconstructed."
        ),
        "key_concepts": [
            "Cosmic ray muon flux (~1 muon/cm²/min at sea level)",
            "Multiple Coulomb scattering",
            "Scattering angle and radiation length",
            "Point of closest approach (POCA) reconstruction",
            "Maximum likelihood / expectation maximisation",
        ],
        "signal_equation": "θ_rms = (13.6 MeV / pβc) · √(x/X₀) · [1 + 0.038 ln(x/X₀)]",
    },
    "Neutron": {
        "intro": (
            "Neutron imaging uses thermal or cold neutrons (wavelength ~0.1-1 nm) "
            "that interact with nuclei rather than electrons. This gives "
            "complementary contrast to X-rays — light elements (H, Li, B) are "
            "strong absorbers, while heavy metals may be transparent."
        ),
        "key_concepts": [
            "Nuclear cross-sections vs photon cross-sections",
            "Complementary contrast to X-rays",
            "Neutron sources: reactors, spallation",
            "Scintillator-based neutron detection",
            "Bragg edge imaging for crystallography",
        ],
        "signal_equation": "I = I₀ · exp(-Σ_t · d) + scatter",
    },
}

# Default fallback
DEFAULT_CARRIER_PHYSICS = {
    "intro": (
        "This modality uses a specific physical probe to interact with the "
        "sample or scene. The interaction produces measurements that encode "
        "information about the object's internal structure or surface properties."
    ),
    "key_concepts": [
        "Probe-sample interaction mechanism",
        "Forward model relating object to measurements",
        "Noise model (signal-dependent and independent)",
        "Spatial resolution and field of view",
        "Contrast mechanism and sensitivity",
    ],
    "signal_equation": "y = A(x) + noise",
}

# ---------------------------------------------------------------------------
# Forward model descriptions
# ---------------------------------------------------------------------------
FORWARD_MODEL_DESCRIPTIONS = {
    "linear_operator": (
        "The forward model is **linear**: y = Ax + n, where A is a linear "
        "operator (matrix or linear transform). This means superposition holds — "
        "doubling the input doubles the output. Many classical reconstruction "
        "algorithms (least-squares, CG, FISTA) exploit linearity."
    ),
    "nonlinear_operator": (
        "The forward model is **nonlinear**: y = f(x) + n, where f is a "
        "nonlinear mapping. This means superposition does not hold, and "
        "iterative linearisation (Newton, Gauss-Newton) or specialised "
        "algorithms are needed for reconstruction."
    ),
    "explicit_matrix": (
        "The forward model uses an **explicit matrix**: y = Φx + n, where "
        "Φ is a known measurement matrix (e.g., random Gaussian, Hadamard). "
        "The matrix is stored and applied directly, enabling compressed "
        "sensing approaches."
    ),
}

CATEGORY_MODULE_DESCRIPTIONS = {
    "medical_ct_radon": "Radon transform / projection-based sensing",
    "medical_mri_kspace": "Fourier / k-space sampling",
    "microscopy_psf": "PSF convolution / deconvolution",
    "compressive_mask": "Coded aperture / compressive sensing",
    "electron_ctf": "Contrast transfer function / electron optics",
    "remote_sensing_sar": "Range-Doppler / synthetic aperture",
    "scanning_probe": "Near-field / tip-sample interaction",
    "nuclear_emission": "Emission tomography / line integrals",
}


def load_modality_config(modality_id: str) -> dict:
    """Load YAML config for a modality."""
    cfg_path = CONFIGS_DIR / f"{modality_id}.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def load_registry_entry(modality_id: str, registry: dict) -> dict:
    """Get entry from modalities.yaml registry."""
    return registry.get(modality_id, {})


def load_solver_entry(modality_id: str, solver_reg: dict) -> dict:
    """Get solver tiers from solver_registry.yaml."""
    return solver_reg.get(modality_id, {})


def get_carrier_physics(carrier: str) -> dict:
    """Get physics primer for a carrier type."""
    return CARRIER_PHYSICS.get(carrier, DEFAULT_CARRIER_PHYSICS)


def format_mismatch_table(params: list) -> str:
    """Format mismatch parameters as markdown table."""
    if not params:
        return "No mismatch parameters defined for this modality.\n"
    lines = ["| Parameter | Nominal | Range | Unit |",
             "|-----------|---------|-------|------|"]
    for p in params:
        name = p.get("name", "unknown")
        nominal = p.get("nominal", "—")
        rng = p.get("range", [])
        unit = p.get("unit", "—")
        if isinstance(rng, list) and len(rng) == 2:
            rng_str = f"{rng[0]} – {rng[1]}"
        else:
            rng_str = str(rng)
        lines.append(f"| {name} | {nominal} | {rng_str} | {unit} |")
    return "\n".join(lines) + "\n"


def format_solver_table(solvers: dict) -> str:
    """Format solver configurations as markdown table."""
    if not solvers:
        return "No solver configurations available.\n"
    lines = ["| Tier | Name | Module | Function | GPU | Reference |",
             "|------|------|--------|----------|-----|-----------|"]
    for tier, info in solvers.items():
        if not isinstance(info, dict):
            continue
        name = info.get("name", "—")
        module = info.get("module", "—")
        func = info.get("function", "—")
        gpu = "Yes" if info.get("gpu", False) else "No"
        ref = info.get("reference", "")
        lines.append(f"| {tier} | {name} | `{module}` | `{func}` | {gpu} | {ref} |")
    return "\n".join(lines) + "\n"


def format_elements_table(elements: list) -> str:
    """Format imaging chain elements."""
    if not elements:
        return ""
    lines = ["\n### Imaging Chain Elements\n",
             "| Element | Type | Transfer | Throughput | Noise |",
             "|---------|------|----------|------------|-------|"]
    for e in elements:
        name = e.get("name", "—")
        etype = e.get("element_type", "—")
        transfer = e.get("transfer_kind", "—")
        throughput = e.get("throughput", "—")
        noise = ", ".join(e.get("noise_kinds", [])) or "—"
        lines.append(f"| {name} | {etype} | {transfer} | {throughput} | {noise} |")
    return "\n".join(lines) + "\n"


def write_file(path: Path, content: str):
    """Write content to file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def generate_readme(cfg: dict, reg: dict, solvers: dict) -> str:
    mod_id = cfg.get("modality_id", "unknown")
    display = cfg.get("display_name", mod_id)
    category = cfg.get("category", "Unknown")
    carrier = cfg.get("carrier", "Unknown")

    return f"""# {display} — Learning Materials

A self-contained curriculum for understanding the physics, forward model,
reconstruction algorithms, and PWM benchmark for **{display}**.

## Quick Facts

| Property | Value |
|----------|-------|
| Modality ID | `{mod_id}` |
| Category | {category} |
| Carrier | {carrier} |
| Forward Model | {cfg.get('forward_model_type', 'unknown')} |
| Default Solver | `{cfg.get('default_solver', 'unknown')}` |
| Maturity | {cfg.get('maturity', 'unknown')} |
| Tier | {cfg.get('tier', 'unknown')} |

## Reading Order

| # | File | Topic | Est. Time |
|---|------|-------|-----------|
| 1 | [01_physics_fundamentals.md](01_physics_fundamentals.md) | Physics of {carrier} imaging, key equations, hardware | 30 min |
| 2 | [02_forward_model.md](02_forward_model.md) | Forward model, mismatch parameters, noise model | 20 min |
| 3 | [03_reconstruction_algorithms.md](03_reconstruction_algorithms.md) | Available solvers, comparison, trade-offs | 25 min |
| 4 | [04_pwm_benchmark.md](04_pwm_benchmark.md) | PWM benchmark structure, data format, scoring | 15 min |
| 5 | [05_hands_on_tutorial.md](05_hands_on_tutorial.md) | Code snippets: load data, run solvers, compute metrics | 20 min |

**Total estimated reading time: ~2 hours**

## Key Source Files

| Module | Path |
|--------|------|
| Benchmark config | `benchmarks/configs/{mod_id}.yaml` |
| Modality registry | `packages/pwm_core/contrib/modalities.yaml` |
| Solver registry | `packages/pwm_core/contrib/solver_registry.yaml` |
"""


def generate_01_physics(cfg: dict, reg: dict) -> str:
    display = cfg.get("display_name", cfg.get("modality_id", "Unknown"))
    carrier = cfg.get("carrier", "Unknown")
    physics = get_carrier_physics(carrier)
    category = cfg.get("category", "Unknown")

    # Get description from registry
    description = reg.get("description", "")
    if not description:
        description = f"{display} is an imaging technique in the {category} domain."

    # Get elements from registry
    elements = reg.get("elements", [])
    elements_table = format_elements_table(elements)

    # Get wavelength info
    wl = reg.get("wavelength_range_nm", cfg.get("wavelength_range_nm"))
    wl_section = ""
    if wl and isinstance(wl, list) and len(wl) == 2:
        wl_section = f"\n### Wavelength / Energy Range\n\n{wl[0]} – {wl[1]} nm\n"

    key_concepts_md = "\n".join(f"- {c}" for c in physics["key_concepts"])

    return f"""# 01 — Physics Fundamentals: {display}

## 1. Overview

{description.strip()}

**Category**: {category}
**Carrier**: {carrier}

---

## 2. {carrier} Physics

{physics['intro']}

### Key Concepts

{key_concepts_md}
{wl_section}
---

## 3. Signal Equation

The fundamental signal equation for this modality:

```
{physics['signal_equation']}
```

This describes how the object (x) produces measurements (y) through the
physical imaging process. The goal of reconstruction is to invert this
relationship.

---

## 4. Hardware and Imaging Chain
{elements_table}
### System Parameters

| Parameter | Value |
|-----------|-------|
| Image shape (x) | {cfg.get('x_shape', 'varies')} |
| Measurement shape (y) | {cfg.get('y_shape', 'varies')} |
| Forward model type | {cfg.get('forward_model_type', 'unknown')} |
| Category module | {cfg.get('category_module', 'unknown')} |

---

## 5. Key Physics Parameters

"""  + _format_theta(cfg.get("theta", {})) + """

---

## 6. Summary

| Aspect | Details |
|--------|---------|
| Physical probe | """ + carrier + """ |
| Primary contrast | Determined by """ + carrier.lower() + """-matter interaction |
| Resolution limit | Set by wavelength / aperture / probe geometry |
| Noise model | Signal-dependent (Poisson/speckle) + signal-independent (Gaussian) |

---

*Next: [02 — Forward Model](02_forward_model.md)*
"""


def _format_theta(theta: dict) -> str:
    if not theta:
        return "No specific physics parameters defined.\n"
    lines = ["| Parameter | Value |", "|-----------|-------|"]
    for k, v in theta.items():
        lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
    return "\n".join(lines) + "\n"


def generate_02_forward_model(cfg: dict, reg: dict) -> str:
    display = cfg.get("display_name", cfg.get("modality_id", "Unknown"))
    carrier = cfg.get("carrier", "Unknown")
    fwd_type = cfg.get("forward_model_type", "unknown")
    cat_module = cfg.get("category_module", "unknown")
    physics = get_carrier_physics(carrier)

    fwd_desc = FORWARD_MODEL_DESCRIPTIONS.get(fwd_type,
        "The forward model maps the object to measurements.")
    cat_desc = CATEGORY_MODULE_DESCRIPTIONS.get(cat_module,
        "Specialised physics engine for this category")

    mismatch = cfg.get("mismatch_params", [])
    mismatch_table = format_mismatch_table(mismatch)

    return f"""# 02 — Forward Model: {display}

## 1. The Forward Model

### 1.1 Model Type: `{fwd_type}`

{fwd_desc}

### 1.2 Signal Equation

```
{physics['signal_equation']}
```

### 1.3 Physics Engine

This modality uses the **`{cat_module}`** category module:
{cat_desc}.

---

## 2. Mismatch Parameters

The PWM benchmark introduces physics mismatch between the true acquisition
and what algorithms assume. This tests algorithm robustness.

### Mismatch Parameter Table

{mismatch_table}

### Mismatch Philosophy

- **Public tier**: mild mismatch — algorithms should perform well
- **Dev tier**: moderate mismatch — exposes fragility
- **Hidden tier**: severe mismatch — tests true robustness

The gap between public and hidden tier performance reveals how sensitive
an algorithm is to model errors.

---

## 3. Noise Model

The noise model combines:

1. **Signal-dependent noise**: Poisson (photon counting), speckle
   (coherent), or multiplicative noise
2. **Signal-independent noise**: Gaussian read noise, dark current,
   thermal noise
3. **Systematic errors**: background, fixed-pattern noise, calibration
   errors

The relative importance depends on the imaging regime:
- Low-signal: Poisson-dominated → shot noise is the bottleneck
- High-signal: calibration-dominated → mismatch is the bottleneck

---

## 4. Why Reconstruction is Challenging

The inverse problem y = A(x) + n is challenging because:

1. **Ill-conditioning**: small changes in y cause large changes in x
2. **Underdetermination**: fewer measurements than unknowns (compressed sensing)
3. **Nonlinearity**: the forward model may be nonlinear
4. **Model mismatch**: A_true ≠ A_assumed
5. **Noise amplification**: regularisation is needed to control noise

---

## 5. Connection to PWM Framework

The PWM benchmark for {display} uses:

- **Forward model**: `{cat_module}` with `{fwd_type}` operator
- **Default solver**: `{cfg.get('default_solver', 'unknown')}`
- **Metrics**: {cfg.get('metrics', {}).get('names', ['psnr', 'ssim'])}
- **Primary metric**: {cfg.get('metrics', {}).get('primary', 'psnr')}

---

*Previous: [01 — Physics Fundamentals](01_physics_fundamentals.md)*
*Next: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
"""


def generate_03_algorithms(cfg: dict, solvers: dict) -> str:
    display = cfg.get("display_name", cfg.get("modality_id", "Unknown"))
    default = cfg.get("default_solver", "unknown")

    solver_table = format_solver_table(
        {**cfg.get("solvers", {}), **solvers}
    )

    # Build detailed sections for each solver tier
    solver_sections = []
    all_solvers = {**cfg.get("solvers", {}), **solvers}
    for tier, info in all_solvers.items():
        if not isinstance(info, dict):
            continue
        name = info.get("name", tier)
        module = info.get("module", "")
        func = info.get("function", "")
        gpu = info.get("gpu", False)
        params = info.get("params", "0")
        ref = info.get("reference", "")

        section = f"""### {tier.replace('_', ' ').title()}: {name}

- **Module**: `{module}`
- **Function**: `{func}`
- **Parameters**: {params}
- **GPU required**: {'Yes' if gpu else 'No'}
"""
        if ref:
            section += f"- **Reference**: {ref}\n"
        solver_sections.append(section)

    sections_md = "\n".join(solver_sections) if solver_sections else \
        "Detailed solver configurations are defined in the benchmark YAML config.\n"

    return f"""# 03 — Reconstruction Algorithms: {display}

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for {display} is **`{default}`**.

---

## 2. Solver Comparison Table

{solver_table}

---

## 3. Solver Details

{sections_md}

---

## 4. Algorithm Selection Guide

| Scenario | Recommended Tier | Why |
|----------|------------------|-----|
| Quick baseline | `traditional_cpu` | Fast, no GPU needed |
| Best quality | `best_quality` | Highest PSNR/SSIM |
| Published benchmark | `famous_dl` | Reproducible, citable |
| Limited GPU memory | `small_gpu` | Fits on consumer GPU |

### General Recommendations

- Start with `traditional_cpu` to establish a baseline
- Compare with `best_quality` to see the achievable improvement
- Use `famous_dl` if you need results comparable to published papers
- Use `small_gpu` if GPU memory is limited (< 6 GB VRAM)

---

## 5. Adding a New Solver

To add a new solver to the benchmark:

1. Implement the solver function in `packages/pwm_core/pwm_core/recon/`
2. Register it in `packages/pwm_core/contrib/solver_registry.yaml`
3. Add the solver tier to the modality config in `benchmarks/configs/{cfg.get('modality_id', 'example')}.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
"""


def generate_04_benchmark(cfg: dict) -> str:
    display = cfg.get("display_name", cfg.get("modality_id", "Unknown"))
    mod_id = cfg.get("modality_id", "unknown")
    x_shape = cfg.get("x_shape", [64, 64])
    y_shape = cfg.get("y_shape", [64, 64])
    metrics = cfg.get("metrics", {})
    metric_names = metrics.get("names", ["psnr", "ssim"])
    primary = metrics.get("primary", "psnr")
    thresholds = metrics.get("thresholds", {})

    data_source = cfg.get("data_source", {})
    dataset_id = data_source.get("dataset_id", "")
    dataset_url = data_source.get("dataset_url", "")
    citation = data_source.get("citation", "")
    fallback = data_source.get("fallback", "generated")
    synth_gen = data_source.get("synthetic_generator", "")
    license_ = data_source.get("license", "")

    mismatch = cfg.get("mismatch_params", [])
    mismatch_table = format_mismatch_table(mismatch)

    return f"""# 04 — PWM Benchmark: {display}

## 1. Overview

The PWM benchmark for **{display}** evaluates reconstruction algorithms
under physics model mismatch using a 3-tier structure with increasing
difficulty.

---

## 2. Three-Tier Structure

| Tier | Mismatch | Purpose |
|------|----------|---------|
| **Public** | Mild | Algorithm development, debugging |
| **Dev** | Moderate | Validation, hyperparameter tuning |
| **Hidden** | Severe | Final evaluation, leaderboard |

---

## 3. Data Format

### Signal Dimensions

| Dimension | Shape |
|-----------|-------|
| Object (x) | {x_shape} |
| Measurements (y) | {y_shape} |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `{dataset_id}` |
| Dataset URL | {dataset_url or '—'} |
| Fallback | `{fallback}` |
| Synthetic generator | `{synth_gen}` |
| Citation | {citation or '—'} |
| License | {license_} |

---

## 4. Mismatch Parameters

{mismatch_table}

Each sample in the benchmark has randomly drawn mismatch values from the
ranges above. The mismatch severity increases from public to hidden tier.

---

## 5. Scoring

### Metrics

| Metric | Primary | Threshold |
|--------|:-------:|-----------|
"""  + "\n".join(
        f"| {m} | {'Yes' if m == primary else 'No'} | {thresholds.get(m, '—')} |"
        for m in metric_names
    ) + f"""

Metrics are computed using `benchmarks/framework/metrics.py`:

```python
from benchmarks.framework.metrics import compute_psnr, compute_ssim

psnr = compute_psnr(x_true, x_hat, max_val=1.0)
ssim = compute_ssim(x_true, x_hat, data_range=1.0)
```

---

## 6. Running the Benchmark

```bash
# Using the expanded config runner
python benchmarks/runners/run_expanded.py --modality {mod_id}

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality {mod_id} --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/{mod_id}.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/{mod_id}_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
"""


def generate_05_tutorial(cfg: dict, solvers: dict) -> str:
    display = cfg.get("display_name", cfg.get("modality_id", "Unknown"))
    mod_id = cfg.get("modality_id", "unknown")
    default_solver = cfg.get("default_solver", "unknown")

    # Get the traditional_cpu solver info
    all_solvers = {**cfg.get("solvers", {}), **solvers}
    trad = all_solvers.get("traditional_cpu", {})
    trad_module = trad.get("module", "pwm_core.recon")
    trad_func = trad.get("function", "reconstruct")
    trad_name = trad.get("name", "Traditional")

    best = all_solvers.get("best_quality", {})
    best_module = best.get("module", "pwm_core.recon")
    best_func = best.get("function", "reconstruct")
    best_name = best.get("name", "Best Quality")

    return f"""# 05 — Hands-On Tutorial: {display}

This tutorial walks through running the PWM benchmark for {display},
from loading data to computing metrics.

## Setup

```python
import sys
from pathlib import Path

ROOT = Path("/home/spiritai/abraham/pwm/production/Physics_World_Model")
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))
sys.path.insert(0, str(ROOT))

import numpy as np
```

---

## 1. Loading the Benchmark Config

```python
import yaml

config_path = ROOT / "benchmarks" / "configs" / "{mod_id}.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

print(f"Modality: {{cfg['display_name']}}")
print(f"Category: {{cfg['category']}}")
print(f"Forward model: {{cfg['forward_model_type']}}")
print(f"Default solver: {{cfg['default_solver']}}")
print(f"Image shape: {{cfg['x_shape']}}")
```

---

## 2. Understanding the Forward Model

```python
# The forward model type determines how measurements relate to the object
fwd_type = cfg["forward_model_type"]
cat_module = cfg["category_module"]

print(f"Forward model type: {{fwd_type}}")
print(f"Category module: {{cat_module}}")

# Mismatch parameters define the physics errors
for p in cfg.get("mismatch_params", []):
    print(f"  {{p['name']}}: nominal={{p['nominal']}}, "
          f"range={{p['range']}}, unit={{p['unit']}}")
```

---

## 3. Running the Default Solver

```python
# Import the traditional CPU solver
try:
    from {trad_module} import {trad_func}
    print("Solver loaded: {trad_name}")
except ImportError:
    print("Solver not available — install required dependencies")
```

---

## 4. Running the Benchmark

```python
# Use the expanded benchmark runner
# This handles data loading, solver execution, and metric computation

# Command-line usage:
# python benchmarks/runners/run_expanded.py --modality {mod_id}

# Or programmatically:
from benchmarks.framework.metrics import compute_psnr, compute_ssim

# After obtaining x_true and x_hat:
# psnr = compute_psnr(x_true, x_hat, max_val=1.0)
# ssim = compute_ssim(x_true, x_hat, data_range=1.0)
# print(f"PSNR: {{psnr:.2f}} dB, SSIM: {{ssim:.4f}}")
```

---

## 5. Comparing Solvers

```python
# Compare traditional vs best quality solver
solvers_to_test = {{
    "{trad_name}": ("{trad_module}", "{trad_func}"),
    "{best_name}": ("{best_module}", "{best_func}"),
}}

# results = {{}}
# for name, (module, func) in solvers_to_test.items():
#     x_hat = run_solver(module, func, y, physics)
#     psnr = compute_psnr(x_true, x_hat, max_val=1.0)
#     ssim = compute_ssim(x_true, x_hat, data_range=1.0)
#     results[name] = {{"psnr": psnr, "ssim": ssim}}
#     print(f"{{name}}: PSNR={{psnr:.2f}}, SSIM={{ssim:.4f}}")
```

---

## 6. Visualising Results

```python
import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(x_true, cmap="gray")
# axes[0].set_title("Ground Truth")
# axes[1].imshow(x_hat_trad, cmap="gray")
# axes[1].set_title("{trad_name}")
# axes[2].imshow(x_hat_best, cmap="gray")
# axes[2].set_title("{best_name}")
# for ax in axes:
#     ax.axis("off")
# plt.tight_layout()
# plt.savefig("{mod_id}_comparison.png", dpi=150)
# plt.show()
```

---

## 7. Next Steps

- Read the full config: `benchmarks/configs/{mod_id}.yaml`
- Explore the expanded config: `benchmarks/expanded_configs/{mod_id}_expanded.yaml`
- Compare across tiers to see mismatch impact
- Add your own solver to the benchmark

---

*Previous: [04 — PWM Benchmark](04_pwm_benchmark.md)*
*Back to: [README](README.md)*
"""


def generate_modality(modality_id: str, registry: dict, solver_reg: dict,
                      skip_existing: bool = False):
    """Generate all learning files for one modality."""
    out_dir = LEARN_DIR / modality_id
    if skip_existing and (out_dir / "README.md").exists():
        return False

    cfg = load_modality_config(modality_id)
    if not cfg:
        print(f"  [SKIP] No config found for {modality_id}")
        return False

    reg = load_registry_entry(modality_id, registry)
    solvers = load_solver_entry(modality_id, solver_reg)

    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "README.md": generate_readme(cfg, reg, solvers),
        "01_physics_fundamentals.md": generate_01_physics(cfg, reg),
        "02_forward_model.md": generate_02_forward_model(cfg, reg),
        "03_reconstruction_algorithms.md": generate_03_algorithms(cfg, solvers),
        "04_pwm_benchmark.md": generate_04_benchmark(cfg),
        "05_hands_on_tutorial.md": generate_05_tutorial(cfg, solvers),
    }

    for fname, content in files.items():
        write_file(out_dir / fname, content)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate learning materials for all PWM modalities")
    parser.add_argument("--modality", type=str, default=None,
                        help="Generate for a single modality (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip modalities that already have README.md")
    parser.add_argument("--skip-mri", action="store_true", default=True,
                        help="Skip MRI (already hand-crafted)")
    args = parser.parse_args()

    # Load registries
    print("Loading modality registry...")
    with open(MODALITIES_YAML) as f:
        registry_raw = yaml.safe_load(f)
    registry = registry_raw.get("modalities", registry_raw)

    print("Loading solver registry...")
    with open(SOLVER_REGISTRY) as f:
        solver_reg = yaml.safe_load(f)

    # Get modality list
    if args.modality:
        modalities = [args.modality]
    else:
        modalities = sorted([
            p.stem for p in CONFIGS_DIR.glob("*.yaml")
            if p.stem != "_template"
        ])

    if args.skip_mri:
        modalities = [m for m in modalities if m != "mri"]

    print(f"Generating learning materials for {len(modalities)} modalities...")

    created = 0
    skipped = 0
    failed = 0

    for i, mod_id in enumerate(modalities, 1):
        print(f"  [{i:3d}/{len(modalities)}] {mod_id}...", end=" ")
        try:
            ok = generate_modality(mod_id, registry, solver_reg,
                                   skip_existing=args.skip_existing)
            if ok:
                print("OK")
                created += 1
            else:
                print("SKIPPED")
                skipped += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nDone: {created} created, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
