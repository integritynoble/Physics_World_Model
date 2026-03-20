#!/usr/bin/env python3
"""Backfill mismatch_params for modalities that have empty params.

Adds physically appropriate mismatch parameters based on each modality's
carrier type, category, and forward model (DAG).
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "configs"

# ── Physics-based mismatch_params for each missing modality ──────────────────

MISMATCH_PARAMS: dict[str, list[dict]] = {
    "acoustic_emission": [
        {"name": "Source location error", "nominal": 0.0, "range": [-5.0, 5.0], "unit": "mm"},
        {"name": "Wave speed error", "nominal": 5900.0, "range": [5700.0, 6100.0], "unit": "m/s"},
        {"name": "Sensor coupling gain", "nominal": 1.0, "range": [0.8, 1.2], "unit": "-"},
        {"name": "Arrival time bias", "nominal": 0.0, "range": [-0.5, 0.5], "unit": "us"},
    ],
    "active_thermography": [
        {"name": "Emissivity error", "nominal": 0.95, "range": [0.85, 1.0], "unit": "-"},
        {"name": "Heat source power drift", "nominal": 1.0, "range": [0.9, 1.1], "unit": "-"},
        {"name": "Background temperature", "nominal": 25.0, "range": [20.0, 30.0], "unit": "C"},
        {"name": "Integration time offset", "nominal": 0.0, "range": [-0.1, 0.1], "unit": "s"},
    ],
    "adaptive_optics": [
        {"name": "DM actuator gain", "nominal": 1.0, "range": [0.9, 1.1], "unit": "-"},
        {"name": "WFS centroid bias", "nominal": 0.0, "range": [-0.2, 0.2], "unit": "px"},
        {"name": "Fried parameter r0", "nominal": 0.15, "range": [0.10, 0.25], "unit": "m"},
        {"name": "Servo lag", "nominal": 0.0, "range": [0.0, 2.0], "unit": "ms"},
    ],
    "atom_probe": [
        {"name": "Flight path error", "nominal": 0.0, "range": [-0.5, 0.5], "unit": "mm"},
        {"name": "Voltage calibration", "nominal": 1.0, "range": [0.98, 1.02], "unit": "-"},
        {"name": "Detection efficiency", "nominal": 0.6, "range": [0.5, 0.7], "unit": "-"},
        {"name": "Tip radius error", "nominal": 0.0, "range": [-5.0, 5.0], "unit": "nm"},
    ],
    "brachytherapy_img": [
        {"name": "Source position error", "nominal": 0.0, "range": [-2.0, 2.0], "unit": "mm"},
        {"name": "Attenuation coefficient", "nominal": 0.2, "range": [0.15, 0.25], "unit": "1/cm"},
        {"name": "Detector gain drift", "nominal": 1.0, "range": [0.95, 1.05], "unit": "-"},
        {"name": "Scatter fraction", "nominal": 0.15, "range": [0.10, 0.25], "unit": "-"},
    ],
    "cryo_em": [
        {"name": "Defocus error", "nominal": 0.0, "range": [-500.0, 500.0], "unit": "nm"},
        {"name": "Astigmatism", "nominal": 0.0, "range": [0.0, 100.0], "unit": "nm"},
        {"name": "Beam tilt", "nominal": 0.0, "range": [-1.0, 1.0], "unit": "mrad"},
        {"name": "Ice thickness variation", "nominal": 50.0, "range": [30.0, 80.0], "unit": "nm"},
    ],
    "eddy_current": [
        {"name": "Liftoff distance", "nominal": 0.0, "range": [0.0, 1.0], "unit": "mm"},
        {"name": "Conductivity error", "nominal": 58.0, "range": [55.0, 61.0], "unit": "MS/m"},
        {"name": "Excitation frequency drift", "nominal": 100.0, "range": [95.0, 105.0], "unit": "kHz"},
        {"name": "Probe tilt", "nominal": 0.0, "range": [-2.0, 2.0], "unit": "deg"},
    ],
    "gpr": [
        {"name": "Soil permittivity error", "nominal": 9.0, "range": [6.0, 15.0], "unit": "-"},
        {"name": "Antenna height", "nominal": 0.0, "range": [-0.05, 0.05], "unit": "m"},
        {"name": "Time zero offset", "nominal": 0.0, "range": [-0.5, 0.5], "unit": "ns"},
        {"name": "Velocity model error", "nominal": 0.1, "range": [0.08, 0.13], "unit": "m/ns"},
    ],
    "gravitational_wave": [
        {"name": "Calibration amplitude", "nominal": 1.0, "range": [0.95, 1.05], "unit": "-"},
        {"name": "Phase calibration", "nominal": 0.0, "range": [-0.05, 0.05], "unit": "rad"},
        {"name": "Power spectral density", "nominal": 1e-23, "range": [5e-24, 2e-23], "unit": "1/Hz"},
        {"name": "Timing offset", "nominal": 0.0, "range": [-1e-4, 1e-4], "unit": "s"},
    ],
    "hyperspectral_remote": [
        {"name": "Spectral shift", "nominal": 0.0, "range": [-2.0, 2.0], "unit": "nm"},
        {"name": "Smile distortion", "nominal": 0.0, "range": [-1.0, 1.0], "unit": "px"},
        {"name": "Keystone distortion", "nominal": 0.0, "range": [-0.5, 0.5], "unit": "px"},
        {"name": "Radiometric gain", "nominal": 1.0, "range": [0.9, 1.1], "unit": "-"},
    ],
    "impedance_tomo": [
        {"name": "Contact impedance", "nominal": 100.0, "range": [50.0, 200.0], "unit": "ohm"},
        {"name": "Electrode position error", "nominal": 0.0, "range": [-2.0, 2.0], "unit": "mm"},
        {"name": "Background conductivity", "nominal": 0.2, "range": [0.1, 0.4], "unit": "S/m"},
        {"name": "Current amplitude drift", "nominal": 1.0, "range": [0.95, 1.05], "unit": "-"},
    ],
    "industrial_ct": [
        {"name": "Center of rotation offset", "nominal": 0.0, "range": [-2.0, 2.0], "unit": "px"},
        {"name": "Source voltage drift", "nominal": 150.0, "range": [145.0, 155.0], "unit": "kV"},
        {"name": "Detector tilt", "nominal": 0.0, "range": [-0.5, 0.5], "unit": "deg"},
        {"name": "Beam hardening coefficient", "nominal": 0.0, "range": [0.0, 0.05], "unit": "-"},
    ],
    "machine_vision": [
        {"name": "Focus distance error", "nominal": 0.0, "range": [-5.0, 5.0], "unit": "mm"},
        {"name": "Lens distortion k1", "nominal": 0.0, "range": [-0.1, 0.1], "unit": "-"},
        {"name": "Exposure time drift", "nominal": 10.0, "range": [8.0, 12.0], "unit": "ms"},
        {"name": "White balance gain", "nominal": 1.0, "range": [0.9, 1.1], "unit": "-"},
    ],
    "magnetic_particle": [
        {"name": "Drive field amplitude", "nominal": 25.0, "range": [22.0, 28.0], "unit": "mT"},
        {"name": "Selection field gradient", "nominal": 2.5, "range": [2.0, 3.0], "unit": "T/m"},
        {"name": "Particle relaxation time", "nominal": 2.0, "range": [1.0, 3.0], "unit": "us"},
        {"name": "Receive coil sensitivity", "nominal": 1.0, "range": [0.85, 1.15], "unit": "-"},
    ],
    "maldi_msi": [
        {"name": "Laser fluence drift", "nominal": 1.0, "range": [0.8, 1.2], "unit": "-"},
        {"name": "Mass accuracy", "nominal": 0.0, "range": [-5.0, 5.0], "unit": "ppm"},
        {"name": "Extraction delay", "nominal": 100.0, "range": [80.0, 120.0], "unit": "ns"},
        {"name": "Matrix crystallization", "nominal": 1.0, "range": [0.7, 1.3], "unit": "-"},
    ],
    "multispectral_sat": [
        {"name": "Band registration error", "nominal": 0.0, "range": [-1.0, 1.0], "unit": "px"},
        {"name": "Atmospheric transmittance", "nominal": 0.85, "range": [0.70, 0.95], "unit": "-"},
        {"name": "Radiometric calibration", "nominal": 1.0, "range": [0.95, 1.05], "unit": "-"},
        {"name": "Pointing jitter", "nominal": 0.0, "range": [-0.5, 0.5], "unit": "px"},
    ],
    "particle_calorimetry": [
        {"name": "Energy scale factor", "nominal": 1.0, "range": [0.97, 1.03], "unit": "-"},
        {"name": "Position resolution", "nominal": 0.0, "range": [0.0, 5.0], "unit": "mm"},
        {"name": "Sampling fraction", "nominal": 0.1, "range": [0.08, 0.12], "unit": "-"},
        {"name": "Pile-up fraction", "nominal": 0.0, "range": [0.0, 0.05], "unit": "-"},
    ],
    "passive_microwave": [
        {"name": "Antenna beam width error", "nominal": 0.0, "range": [-0.5, 0.5], "unit": "deg"},
        {"name": "Receiver gain drift", "nominal": 1.0, "range": [0.95, 1.05], "unit": "-"},
        {"name": "Brightness temperature offset", "nominal": 0.0, "range": [-2.0, 2.0], "unit": "K"},
        {"name": "Cross-polarization leakage", "nominal": 0.0, "range": [0.0, 0.02], "unit": "-"},
    ],
    "portal_imaging": [
        {"name": "Isocenter shift", "nominal": 0.0, "range": [-2.0, 2.0], "unit": "mm"},
        {"name": "Beam energy variation", "nominal": 6.0, "range": [5.8, 6.2], "unit": "MV"},
        {"name": "Detector sag", "nominal": 0.0, "range": [-1.0, 1.0], "unit": "mm"},
        {"name": "Scatter kernel width", "nominal": 5.0, "range": [3.0, 7.0], "unit": "mm"},
    ],
    "proton_therapy_img": [
        {"name": "Range uncertainty", "nominal": 0.0, "range": [-3.0, 3.0], "unit": "mm"},
        {"name": "Scattering power error", "nominal": 1.0, "range": [0.95, 1.05], "unit": "-"},
        {"name": "Detector efficiency drift", "nominal": 0.85, "range": [0.80, 0.90], "unit": "-"},
        {"name": "Setup error", "nominal": 0.0, "range": [-2.0, 2.0], "unit": "mm"},
    ],
    "radio_astronomy": [
        {"name": "Antenna gain error", "nominal": 1.0, "range": [0.95, 1.05], "unit": "-"},
        {"name": "Phase calibration error", "nominal": 0.0, "range": [-5.0, 5.0], "unit": "deg"},
        {"name": "Bandpass slope", "nominal": 0.0, "range": [-0.01, 0.01], "unit": "1/MHz"},
        {"name": "Pointing offset", "nominal": 0.0, "range": [-5.0, 5.0], "unit": "arcsec"},
    ],
    "radio_interferometry": [
        {"name": "Baseline error", "nominal": 0.0, "range": [-0.01, 0.01], "unit": "m"},
        {"name": "Phase calibration", "nominal": 0.0, "range": [-10.0, 10.0], "unit": "deg"},
        {"name": "Amplitude calibration", "nominal": 1.0, "range": [0.9, 1.1], "unit": "-"},
        {"name": "Clock offset", "nominal": 0.0, "range": [-1.0, 1.0], "unit": "ns"},
    ],
    "saxs": [
        {"name": "Sample-detector distance", "nominal": 1000.0, "range": [990.0, 1010.0], "unit": "mm"},
        {"name": "Beam center x", "nominal": 0.0, "range": [-2.0, 2.0], "unit": "px"},
        {"name": "Beam center y", "nominal": 0.0, "range": [-2.0, 2.0], "unit": "px"},
        {"name": "Wavelength error", "nominal": 0.0, "range": [-0.001, 0.001], "unit": "nm"},
    ],
    "seismic_tomo": [
        {"name": "Velocity model error", "nominal": 5000.0, "range": [4500.0, 5500.0], "unit": "m/s"},
        {"name": "Source location error", "nominal": 0.0, "range": [-50.0, 50.0], "unit": "m"},
        {"name": "Receiver coupling", "nominal": 1.0, "range": [0.85, 1.15], "unit": "-"},
        {"name": "Timing error", "nominal": 0.0, "range": [-0.002, 0.002], "unit": "s"},
    ],
    "terahertz": [
        {"name": "Pulse chirp", "nominal": 0.0, "range": [-0.1, 0.1], "unit": "ps^2"},
        {"name": "Water vapor absorption", "nominal": 0.1, "range": [0.0, 0.3], "unit": "1/cm"},
        {"name": "Beam alignment error", "nominal": 0.0, "range": [-1.0, 1.0], "unit": "mm"},
        {"name": "Dynamic range drift", "nominal": 1.0, "range": [0.9, 1.1], "unit": "-"},
    ],
    "ultrasonic_phased_array": [
        {"name": "Element pitch error", "nominal": 0.0, "range": [-0.01, 0.01], "unit": "mm"},
        {"name": "Sound velocity", "nominal": 5900.0, "range": [5700.0, 6100.0], "unit": "m/s"},
        {"name": "Wedge angle error", "nominal": 0.0, "range": [-1.0, 1.0], "unit": "deg"},
        {"name": "Dead element fraction", "nominal": 0.0, "range": [0.0, 0.05], "unit": "-"},
    ],
    "weather_radar": [
        {"name": "Calibration bias", "nominal": 0.0, "range": [-1.0, 1.0], "unit": "dBZ"},
        {"name": "Beam elevation error", "nominal": 0.0, "range": [-0.2, 0.2], "unit": "deg"},
        {"name": "Attenuation correction", "nominal": 1.0, "range": [0.9, 1.1], "unit": "-"},
        {"name": "Ground clutter leakage", "nominal": 0.0, "range": [0.0, 0.05], "unit": "-"},
    ],
    "xray_crystallography": [
        {"name": "Crystal orientation error", "nominal": 0.0, "range": [-0.5, 0.5], "unit": "deg"},
        {"name": "Beam divergence", "nominal": 0.1, "range": [0.05, 0.2], "unit": "mrad"},
        {"name": "Absorption correction", "nominal": 1.0, "range": [0.9, 1.1], "unit": "-"},
        {"name": "Scale factor", "nominal": 1.0, "range": [0.95, 1.05], "unit": "-"},
    ],
    "xray_ndt": [
        {"name": "Source position error", "nominal": 0.0, "range": [-1.0, 1.0], "unit": "mm"},
        {"name": "Beam hardening", "nominal": 0.0, "range": [0.0, 0.1], "unit": "-"},
        {"name": "Detector gain drift", "nominal": 1.0, "range": [0.95, 1.05], "unit": "-"},
        {"name": "Geometric magnification", "nominal": 2.0, "range": [1.8, 2.2], "unit": "-"},
    ],
    "xrf_imaging": [
        {"name": "Excitation energy drift", "nominal": 0.0, "range": [-0.05, 0.05], "unit": "keV"},
        {"name": "Detector resolution", "nominal": 130.0, "range": [120.0, 150.0], "unit": "eV"},
        {"name": "Matrix absorption", "nominal": 1.0, "range": [0.85, 1.15], "unit": "-"},
        {"name": "Beam spot size", "nominal": 1.0, "range": [0.5, 2.0], "unit": "um"},
    ],
}


def main() -> None:
    for mod_id, params in sorted(MISMATCH_PARAMS.items()):
        yaml_path = CONFIGS_DIR / f"{mod_id}.yaml"
        if not yaml_path.exists():
            print(f"  SKIP: {yaml_path.name} not found")
            continue

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if data.get("mismatch_params"):
            print(f"  SKIP: {mod_id} already has mismatch_params")
            continue

        data["mismatch_params"] = params

        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"  OK: {mod_id} — added {len(params)} mismatch_params")

    print(f"\nDone. Updated {len(MISMATCH_PARAMS)} YAML configs.")
    print("Now run: python platform/scripts/generate_modality_catalog.py")


if __name__ == "__main__":
    main()
