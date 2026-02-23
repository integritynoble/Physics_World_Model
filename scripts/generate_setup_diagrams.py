#!/usr/bin/env python3
"""
Generate experimental setup schematic diagrams for all 64 PWM modalities.

Produces 800×400 px PNG block/flow diagrams showing each modality's physical
signal chain (Source → Optics → Sample → Detector → Reconstruction).
Blocks are colour-coded by component type.

Output directory: platform/pwm_platform/static/img/setups/<key>.png

Usage:
    python scripts/generate_setup_diagrams.py
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ── project root ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "platform"))

from pwm_platform.services.modality_database import MODALITY_DATABASE  # noqa: E402

# ── output directory ──────────────────────────────────────────────────────
OUT_DIR = ROOT / "platform" / "pwm_platform" / "static" / "img" / "setups"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette by component role ──────────────────────────────────────
ROLE_COLORS = {
    "source": "#4A90D9",       # blue
    "optics": "#50B86C",       # green
    "sample": "#E8943A",       # orange
    "detector": "#D94A4A",     # red
    "computation": "#8B5CF6",  # purple
    "encoding": "#0EA5E9",     # sky blue
    "field": "#14B8A6",        # teal
}

# ── Pipeline definitions per category / physics class ─────────────────────
# Each entry is a list of (label, detail_key_or_static, role) tuples.
# detail_key_or_static: if it exists as a key in experimental_setup, use that
# value; otherwise treat as a literal string.


def _get(setup: dict, key: str, fallback: str = "") -> str:
    """Try to pull a value from the experimental_setup dict."""
    val = setup.get(key, fallback)
    if isinstance(val, (int, float)):
        return str(val)
    return str(val) if val else fallback


def _wrap(text: str, width: int = 18) -> str:
    """Wrap text to fit inside diagram boxes."""
    return "\n".join(textwrap.wrap(str(text), width=width))


def _pipeline_for_modality(key: str, info: dict) -> list[tuple[str, str, str]]:
    """Return the signal-chain pipeline as [(label, detail, role), ...]."""
    cat = info.get("category", "")
    setup = info.get("experimental_setup", {})
    physics = info.get("physics_class", "")
    source_type = info.get("source_type", "")
    sensor_type = info.get("sensor_type", "")

    # ── MICROSCOPY ────────────────────────────────────────────────────────
    if cat == "microscopy":
        source_detail = _get(setup, "excitation_source",
                             _get(setup, "illumination",
                                  _get(setup, "source", source_type)))
        optics_detail = _get(setup, "objective",
                             _get(setup, "optics", "Objective lens"))
        sample_detail = "Specimen"
        if "pinhole_AU" in setup:
            optics_detail += f"\nPinhole {setup['pinhole_AU']} AU"
        filter_detail = _get(setup, "emission_filter",
                             _get(setup, "dichroic", "Emission filter"))
        det_detail = _get(setup, "detector", sensor_type)
        recon_detail = _get(setup, "reconstruction", "Reconstruction")

        pipeline = [
            ("Source", source_detail, "source"),
            ("Objective", optics_detail, "optics"),
            ("Sample", sample_detail, "sample"),
            ("Filter", filter_detail, "optics"),
            ("Detector", det_detail, "detector"),
            ("Reconstruction", recon_detail, "computation"),
        ]

        # special: SIM has pattern generation
        if key == "sim":
            pipeline.insert(1, ("Pattern\nGenerator", _get(setup, "pattern_generation",
                                                           "Structured illumination"), "encoding"))
        # special: FLIM has time-correlated detection
        if key == "flim":
            pipeline[4] = ("TCSPC\nDetector", det_detail, "detector")
        # special: FPM has LED array
        if key == "fpm":
            pipeline[0] = ("LED Array", source_detail, "source")
            pipeline[1] = ("Low-NA\nObjective", optics_detail, "optics")

        return pipeline

    # ── COMPRESSIVE ───────────────────────────────────────────────────────
    if cat == "compressive":
        source_detail = _get(setup, "excitation_source",
                             _get(setup, "illumination",
                                  _get(setup, "light_source", source_type)))
        encoding_detail = _get(setup, "coded_aperture",
                               _get(setup, "mask",
                                    _get(setup, "dmd", "Coded aperture")))
        sample_detail = _get(setup, "scene", "Scene / Sample")
        det_detail = _get(setup, "detector", sensor_type)
        recon_detail = _get(setup, "reconstruction", "CS Reconstruction")

        pipeline = [
            ("Source", source_detail, "source"),
            ("Encoding", encoding_detail, "encoding"),
            ("Sample", sample_detail, "sample"),
            ("Detector", det_detail, "detector"),
            ("CS Recon", recon_detail, "computation"),
        ]

        # CACTI: temporal encoding
        if key == "cacti":
            pipeline[1] = ("Temporal\nMask", encoding_detail, "encoding")
        # SPC: single pixel
        if key == "spc":
            pipeline[3] = ("Single\nPixel Det.", det_detail, "detector")

        return pipeline

    # ── MEDICAL ───────────────────────────────────────────────────────────
    if cat == "medical":
        # MRI family
        if physics in ("fourier_sampling", "diffusion_weighted",
                       "spectroscopy", "bold_fmri") or key in (
                           "mri", "fmri", "mrs", "diffusion_mri"):
            return [
                ("B₀ + RF", _get(setup, "instrument", "MRI Scanner"), "source"),
                ("Gradients", _get(setup, "sequence",
                                   _get(setup, "gradient_mode", "Encoding gradients")), "encoding"),
                ("Patient", _get(setup, "anatomy", "Anatomy"), "sample"),
                ("Coil Array", f"{_get(setup, 'receive_coils', '?')} ch", "detector"),
                ("k-space\nRecon", _get(setup, "reconstruction", "IFFT / SENSE"), "computation"),
            ]

        # CT family
        if physics == "tomographic" or key in ("ct", "cbct"):
            return [
                ("X-ray\nTube", _get(setup, "kVp",
                                     _get(setup, "voltage", "X-ray source")) + " kVp"
                 if "kVp" in setup else _get(setup, "instrument", "X-ray tube"), "source"),
                ("Collimator", _get(setup, "dose_level", "Beam shaping"), "optics"),
                ("Patient", "Body / Phantom", "sample"),
                ("Detector\nArray", _get(setup, "detector_pixels",
                                         _get(setup, "detector", "Detector")) + " px"
                 if "detector_pixels" in setup else _get(setup, "detector", "Flat panel"), "detector"),
                ("FBP / Iter.", _get(setup, "reconstruction", "Reconstruction"), "computation"),
            ]

        # X-ray projection (radiography, fluoroscopy, mammography, DEXA, angiography)
        if physics in ("projection", "x_ray_projection", "attenuation") or key in (
                "xray_radiography", "fluoroscopy", "mammography", "dexa", "angiography"):
            return [
                ("X-ray\nSource", _get(setup, "instrument",
                                       _get(setup, "xray_source", "X-ray tube")), "source"),
                ("Filtration", _get(setup, "filter",
                                    _get(setup, "collimation", "Al / Cu filter")), "optics"),
                ("Patient", _get(setup, "anatomy", "Anatomy"), "sample"),
                ("Detector", _get(setup, "detector",
                                  _get(setup, "image_size", "Digital detector")), "detector"),
                ("Processing", _get(setup, "reconstruction",
                                    _get(setup, "processing", "Image processing")), "computation"),
            ]

        # Ultrasound family
        if key in ("ultrasound", "doppler_ultrasound", "elastography"):
            return [
                ("Transducer", _get(setup, "transducer",
                                    _get(setup, "probe", "US Probe")), "source"),
                ("Tx Beam", _get(setup, "transmit_mode",
                                 _get(setup, "mode", "Focused / PW")), "encoding"),
                ("Tissue", _get(setup, "anatomy",
                                _get(setup, "target", "Tissue")), "sample"),
                ("Rx Array", _get(setup, "detector",
                                  _get(setup, "receive_channels", "Receive array")), "detector"),
                ("Beamform", _get(setup, "reconstruction",
                                  _get(setup, "beamforming", "DAS beamforming")), "computation"),
            ]

        # Photoacoustic / DOT / nuclear (PET, SPECT)
        if key == "photoacoustic":
            return [
                ("Pulsed\nLaser", _get(setup, "laser",
                                       _get(setup, "excitation_source", "Nd:YAG")), "source"),
                ("Optical\nFiber", "Light delivery", "optics"),
                ("Tissue", _get(setup, "anatomy", "Tissue"), "sample"),
                ("US Array", _get(setup, "detector", "Transducer array"), "detector"),
                ("Recon", _get(setup, "reconstruction", "Back-projection"), "computation"),
            ]

        if key == "dot":
            return [
                ("NIR\nSource", _get(setup, "source",
                                     _get(setup, "instrument", "NIR laser/LED")), "source"),
                ("Fiber\nBundle", _get(setup, "fibers", "Source-detector fibers"), "optics"),
                ("Tissue", _get(setup, "anatomy", "Head / Breast"), "sample"),
                ("Photo-\ndetector", _get(setup, "detector", "APD / SiPM"), "detector"),
                ("Diffuse\nRecon", _get(setup, "reconstruction", "Iterative DOT"), "computation"),
            ]

        if key in ("pet", "spect"):
            tracer = "Radiotracer"
            if key == "pet":
                return [
                    ("Radio-\ntracer", _get(setup, "tracer",
                                            _get(setup, "radiopharmaceutical", tracer)), "source"),
                    ("Patient", _get(setup, "anatomy", "Body"), "sample"),
                    ("Detector\nRing", _get(setup, "detector", "Scintillator ring"), "detector"),
                    ("Coincidence", "LOR detection", "encoding"),
                    ("OSEM\nRecon", _get(setup, "reconstruction", "MLEM / OSEM"), "computation"),
                ]
            else:
                return [
                    ("Radio-\ntracer", _get(setup, "tracer",
                                            _get(setup, "radiopharmaceutical", tracer)), "source"),
                    ("Patient", _get(setup, "anatomy", "Body"), "sample"),
                    ("Gamma\nCamera", _get(setup, "detector", "NaI(Tl) detector"), "detector"),
                    ("Collimator", _get(setup, "collimator", "Parallel-hole"), "optics"),
                    ("OSEM\nRecon", _get(setup, "reconstruction", "FBP / OSEM"), "computation"),
                ]

        # fallback medical
        return [
            ("Source", _get(setup, "instrument", source_type), "source"),
            ("Optics", "Beam path", "optics"),
            ("Patient", "Anatomy", "sample"),
            ("Detector", _get(setup, "detector", sensor_type), "detector"),
            ("Recon", _get(setup, "reconstruction", "Reconstruction"), "computation"),
        ]

    # ── COHERENT ──────────────────────────────────────────────────────────
    if cat == "coherent":
        return [
            ("Coherent\nSource", _get(setup, "source",
                                      _get(setup, "illumination",
                                           _get(setup, "laser", "Laser"))), "source"),
            ("Optics", _get(setup, "objective",
                            _get(setup, "optics", "Lens system")), "optics"),
            ("Sample", _get(setup, "specimen",
                            _get(setup, "sample", "Specimen")), "sample"),
            ("Detector", _get(setup, "detector",
                              _get(setup, "camera", sensor_type)), "detector"),
            ("Phase\nRetrieval", _get(setup, "reconstruction",
                                      _get(setup, "algorithm", "Phase retrieval")), "computation"),
        ]

    # ── NEURAL RENDERING ──────────────────────────────────────────────────
    if cat == "neural_rendering":
        return [
            ("Multi-view\nCapture", _get(setup, "training_views", "N views") + " views"
             if "training_views" in setup else "RGB Images", "source"),
            ("Camera\nPoses", _get(setup, "pose_estimation", "COLMAP"), "encoding"),
            ("Scene", _get(setup, "scene_type", "3D Scene"), "sample"),
            ("Neural\nNetwork", _get(setup, "architecture", "MLP / 3DGS"), "computation"),
            ("Novel\nViews", _get(setup, "evaluation", "Rendered output"), "computation"),
        ]

    # ── COMPUTATIONAL ─────────────────────────────────────────────────────
    if cat == "computational":
        return [
            ("Source", _get(setup, "source",
                            _get(setup, "illumination", source_type)), "source"),
            ("Optics", _get(setup, "optics",
                            _get(setup, "lenslet_array",
                                 _get(setup, "objective", "Optical system"))), "optics"),
            ("Scene", _get(setup, "scene", "Scene"), "sample"),
            ("Sensor", _get(setup, "detector",
                            _get(setup, "camera", sensor_type)), "detector"),
            ("Compute", _get(setup, "reconstruction",
                             _get(setup, "algorithm", "Computational recon")), "computation"),
        ]

    # ── CLINICAL OPTICS ───────────────────────────────────────────────────
    if cat == "clinical_optics":
        return [
            ("Light\nSource", _get(setup, "source",
                                   _get(setup, "light_source",
                                        _get(setup, "sld", "SLD / LED"))), "source"),
            ("Optics", _get(setup, "optics",
                            _get(setup, "objective",
                                 _get(setup, "interferometer", "Optical path"))), "optics"),
            ("Eye /\nTissue", _get(setup, "anatomy",
                                   _get(setup, "target", "Retina / Tissue")), "sample"),
            ("Detector", _get(setup, "detector",
                              _get(setup, "camera", sensor_type)), "detector"),
            ("Processing", _get(setup, "reconstruction",
                                _get(setup, "processing", "Image processing")), "computation"),
        ]

    # ── ELECTRON MICROSCOPY ───────────────────────────────────────────────
    if cat == "electron_microscopy":
        return [
            ("Electron\nGun", _get(setup, "accelerating_voltage_kV", "") + " kV"
             if "accelerating_voltage_kV" in setup
             else _get(setup, "source", "FEG"), "source"),
            ("EM Lenses", _get(setup, "condenser",
                               _get(setup, "optics", "Condenser + Objective")), "optics"),
            ("Specimen", _get(setup, "specimen",
                              _get(setup, "sample", "Thin section")), "sample"),
            ("Detector", _get(setup, "detector",
                              _get(setup, "camera", sensor_type)), "detector"),
            ("Analysis", _get(setup, "reconstruction",
                              _get(setup, "processing", "Image / Diffraction analysis")), "computation"),
        ]

    # ── DEPTH IMAGING ─────────────────────────────────────────────────────
    if cat == "depth_imaging":
        return [
            ("Emitter", _get(setup, "source",
                             _get(setup, "laser",
                                  _get(setup, "projector", source_type))), "source"),
            ("Optics", _get(setup, "optics", "Projection optics"), "optics"),
            ("Scene", _get(setup, "scene", "3D Scene"), "sample"),
            ("Sensor", _get(setup, "detector",
                            _get(setup, "camera", sensor_type)), "detector"),
            ("Depth\nCompute", _get(setup, "reconstruction",
                                    _get(setup, "algorithm", "Depth reconstruction")), "computation"),
        ]

    # ── REMOTE SENSING ────────────────────────────────────────────────────
    if cat == "remote_sensing":
        return [
            ("Transmitter", _get(setup, "instrument",
                                 _get(setup, "source", source_type)), "source"),
            ("Propagation", _get(setup, "mode",
                                 _get(setup, "frequency_band", "Signal propagation")), "encoding"),
            ("Target", _get(setup, "target",
                            _get(setup, "scene", "Earth surface / Target")), "sample"),
            ("Receiver", _get(setup, "detector",
                              _get(setup, "receiver", sensor_type)), "detector"),
            ("Processing", _get(setup, "reconstruction",
                                _get(setup, "processing", "Signal processing")), "computation"),
        ]

    # ── PARTICLE IMAGING ──────────────────────────────────────────────────
    if cat == "particle_imaging":
        return [
            ("Source", _get(setup, "source",
                            _get(setup, "beam",
                                 _get(setup, "instrument", source_type))), "source"),
            ("Beam\nShaping", _get(setup, "collimation",
                                   _get(setup, "optics", "Collimation")), "optics"),
            ("Object", _get(setup, "target",
                            _get(setup, "sample", "Target object")), "sample"),
            ("Detector", _get(setup, "detector", sensor_type), "detector"),
            ("Tomo\nRecon", _get(setup, "reconstruction", "Reconstruction"), "computation"),
        ]

    # ── GENERIC FALLBACK ──────────────────────────────────────────────────
    return [
        ("Source", _get(setup, "instrument", source_type), "source"),
        ("Optics", "Beam path", "optics"),
        ("Sample", "Specimen", "sample"),
        ("Detector", _get(setup, "detector", sensor_type), "detector"),
        ("Recon", _get(setup, "reconstruction", "Reconstruction"), "computation"),
    ]


# ── Drawing ───────────────────────────────────────────────────────────────


def draw_diagram(key: str, info: dict, out_path: Path) -> None:
    """Draw and save one modality diagram."""
    pipeline = _pipeline_for_modality(key, info)
    n = len(pipeline)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-1.5, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    display_name = info.get("display_name", key)
    physics_class = info.get("physics_class", "").replace("_", " ")
    ax.text(
        n / 2, 2.5, display_name,
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#1F2937",
    )
    ax.text(
        n / 2, 2.1, physics_class,
        ha="center", va="center", fontsize=9, fontstyle="italic",
        color="#6B7280",
    )

    box_w = 0.72
    box_h = 0.65
    spacing = 1.0
    start_x = (n - 1) * spacing / 2
    y_center = 0.8

    for i, (label, detail, role) in enumerate(pipeline):
        x = i * spacing + (n * spacing / 2 - start_x) / 2
        # Centre horizontally
        x = 0.5 + i * (n / (n + 0.3))
        color = ROLE_COLORS.get(role, "#9CA3AF")

        # Draw rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y_center - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.92,
        )
        ax.add_patch(rect)

        # Label inside box
        ax.text(
            x, y_center + 0.05, _wrap(label, 12),
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color="white", linespacing=1.1,
        )

        # Detail below box
        detail_text = _wrap(str(detail)[:60], 16) if detail else ""
        ax.text(
            x, y_center - box_h / 2 - 0.2, detail_text,
            ha="center", va="top", fontsize=5.5,
            color="#4B5563", linespacing=1.05,
        )

        # Arrow to next box
        if i < n - 1:
            x_next = 0.5 + (i + 1) * (n / (n + 0.3))
            arrow = FancyArrowPatch(
                (x + box_w / 2 + 0.02, y_center),
                (x_next - box_w / 2 - 0.02, y_center),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#9CA3AF",
                linewidth=1.5,
            )
            ax.add_patch(arrow)

    # Legend at bottom
    legend_roles = []
    seen = set()
    for _, _, role in pipeline:
        if role not in seen:
            seen.add(role)
            legend_roles.append(role)

    legend_y = -1.1
    legend_start_x = n / 2 - len(legend_roles) * 0.7 / 2
    for i, role in enumerate(legend_roles):
        lx = legend_start_x + i * 0.85
        color = ROLE_COLORS.get(role, "#9CA3AF")
        ax.add_patch(mpatches.Rectangle(
            (lx - 0.12, legend_y - 0.08), 0.22, 0.16,
            facecolor=color, edgecolor="none", alpha=0.85,
        ))
        ax.text(
            lx + 0.15, legend_y, role.replace("_", " ").title(),
            ha="left", va="center", fontsize=5.5, color="#6B7280",
        )

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight", dpi=100, facecolor="white")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    count = 0
    for key, info in MODALITY_DATABASE.items():
        out_path = OUT_DIR / f"{key}.png"
        try:
            draw_diagram(key, info, out_path)
            count += 1
            print(f"  ✓ {key:30s} → {out_path.name}")
        except Exception as exc:
            print(f"  ✗ {key:30s} — ERROR: {exc}")

    print(f"\nGenerated {count}/{len(MODALITY_DATABASE)} diagrams in {OUT_DIR}")


if __name__ == "__main__":
    main()
