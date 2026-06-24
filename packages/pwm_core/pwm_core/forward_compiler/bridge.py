"""Bridge: build ForwardModels from modality templates or digital-twin specs.

The agent's NL/equation reasoning produces a modality name + dimensions; this
bridge turns that into the concrete primitive pipeline (for CASSI) or a
native_operator stage that delegates to an existing pwm_core.physics operator.

Supported modalities
--------------------
Primitive-IR (fully differentiable, adjoint-checked):
  cassi          — coded-aperture snapshot spectral imaging

Native-operator (wraps pwm_core.physics, forward + adjoint):
  mri            — MRI k-space undersampling (2-D)
  ct             — CT parallel-beam Radon transform (2-D)
  lensless       — lensless camera (Fresnel propagation, 2-D)
  holography     — in-line holography (2-D)
  ptychography   — ptychographic CDI (2-D)
  fluorescence   — fluorescence widefield/confocal microscopy (2-D)
  lightsheet     — light-sheet / SPIM (3-D, x_shape = H×W×D)
  ultrasound     — pulse-echo ultrasound B-mode (2-D)
  photoacoustic  — photoacoustic tomography (2-D)

Usage
-----
  fm = from_modality("mri", H=64, W=64, sampling_rate=0.3)
  op = compile_model(fm)   # → NativeCompiledOperator
  y  = op.forward(x)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pwm_core.forward_compiler.ir import ForwardModel, Stage

# ── helpers ───────────────────────────────────────────────────────────────────

def _native_stage(cls_path: str, **kwargs: Any) -> Stage:
    """Build a single native_operator Stage referencing the given class."""
    return Stage(op="native_operator",
                 params={"class": cls_path, "kwargs": kwargs})


def _native_model(name: str, model_shape, cls_path: str,
                  metadata: Dict[str, Any], **op_kwargs: Any) -> ForwardModel:
    """Build a single-stage native_operator ForwardModel.

    model_shape is the ForwardModel x_shape. op_kwargs are forwarded verbatim
    to the operator constructor (they may include their own x_shape key).
    """
    return ForwardModel(
        name=name,
        x_shape=tuple(int(d) for d in model_shape),
        stages=[_native_stage(cls_path, **op_kwargs)],
        metadata=metadata,
    )


# ── public API ────────────────────────────────────────────────────────────────

def from_modality(modality: str, *, H: int, W: int,
                  N_bands: int = 1,
                  D: int = 1,
                  # CASSI
                  mask: Optional[np.ndarray] = None,
                  dispersion: Optional[Dict[str, Any]] = None,
                  # MRI
                  sampling_rate: float = 0.25,
                  seed: int = 42,
                  # CT
                  n_angles: int = 180,
                  # fluorescence / confocal
                  psf_sigma_ex: float = 1.5,
                  psf_sigma_em: float = 2.0,
                  quantum_yield: float = 0.8,
                  background: float = 0.05,
                  # lensless
                  psf_sigma: float = 10.0,
                  # holography
                  carrier_freq: float = 0.2,
                  reference_amplitude: float = 1.0,
                  # ptychography
                  probe_size: int = 16,
                  n_positions: int = 20,
                  # ultrasound
                  n_elements: int = 32,
                  n_samples: int = 128,
                  speed_of_sound: float = 1540.0,
                  element_pitch: float = 3e-4,
                  fs: float = 40e6,
                  # photoacoustic
                  n_transducers: int = 32,
                  pa_speed_of_sound: float = 1.0,
                  ) -> ForwardModel:
    """Return a ForwardModel for a known modality template.

    Parameters
    ----------
    modality : str
        One of: cassi, mri, ct, lensless, holography, ptychography,
        fluorescence, lightsheet, ultrasound, photoacoustic.
    H, W : int
        Spatial dimensions of the object (rows × cols). For lightsheet,
        also pass D (depth slices).
    """
    modality = modality.lower().replace("-", "_").replace(" ", "_")

    # ── CASSI (primitive-IR) ──────────────────────────────────────────────────
    if modality == "cassi":
        if mask is None:
            mask = np.ones((H, W), dtype=np.float64)
        disp = dispersion or {
            "dispersion_model": "poly",
            "disp_poly_x": [0.0, 1.0],
            "disp_poly_y": [0.0, 0.0],
        }
        return ForwardModel(
            name=f"cassi_{H}x{W}x{N_bands}",
            x_shape=(H, W, N_bands),
            stages=[
                Stage(op="band_shift", params={"dispersion": disp}),
                Stage(op="mask_multiply",
                      params={"mask": np.asarray(mask, dtype=np.float64)}),
                Stage(op="band_sum", params={}),
            ],
            metadata={"modality": "cassi"},
        )

    # ── MRI ───────────────────────────────────────────────────────────────────
    if modality == "mri":
        return _native_model(
            f"mri_{H}x{W}", (H, W),
            "pwm_core.physics.mri.mri_operator.MRIOperator",
            {"modality": "mri"},
            x_shape=[H, W], sampling_rate=sampling_rate, seed=seed,
        )

    # ── CT ───────────────────────────────────────────────────────────────────
    if modality in ("ct", "tomography"):
        return _native_model(
            f"ct_{H}x{W}_{n_angles}angles", (H, W),
            "pwm_core.physics.tomography.ct_operator.CTOperator",
            {"modality": "ct"},
            x_shape=[H, W], n_angles=n_angles,
        )

    # ── Lensless camera (diffuser / Fresnel) ──────────────────────────────────
    if modality in ("lensless", "fresnel"):
        return _native_model(
            f"lensless_{H}x{W}", (H, W),
            "pwm_core.physics.lensless.lensless_operator.LenslessOperator",
            {"modality": "lensless"},
            x_shape=[H, W], psf_sigma=psf_sigma,
        )

    # ── In-line holography ────────────────────────────────────────────────────
    if modality in ("holography", "inline_holography"):
        return _native_model(
            f"holography_{H}x{W}", (H, W),
            "pwm_core.physics.microscopy.holography_operator.HolographyOperator",
            {"modality": "holography"},
            x_shape=[H, W], carrier_freq=carrier_freq,
            reference_amplitude=reference_amplitude,
        )

    # ── Ptychography ──────────────────────────────────────────────────────────
    if modality in ("ptychography", "ptycho"):
        return _native_model(
            f"ptychography_{H}x{W}", (H, W),
            "pwm_core.physics.microscopy.ptychography_operator.PtychographyOperator",
            {"modality": "ptychography"},
            x_shape=[H, W], probe_size=probe_size, n_positions=n_positions,
        )

    # ── Fluorescence / widefield / confocal ───────────────────────────────────
    if modality in ("fluorescence", "widefield", "confocal"):
        return _native_model(
            f"fluorescence_{H}x{W}", (H, W),
            "pwm_core.physics.microscopy.fluorescence_operator."
            "FluorescenceMicroscopyOperator",
            {"modality": "fluorescence"},
            nx=W, ny=H,
            psf_sigma_ex=psf_sigma_ex, psf_sigma_em=psf_sigma_em,
            quantum_yield=quantum_yield, background=background,
        )

    # ── Light-sheet / SPIM (3-D) ──────────────────────────────────────────────
    if modality in ("lightsheet", "spim", "light_sheet"):
        return _native_model(
            f"lightsheet_{H}x{W}x{D}", (H, W, D),
            "pwm_core.physics.microscopy.lightsheet_operator.LightsheetOperator",
            {"modality": "lightsheet"},
            x_shape=[H, W, D],
        )

    # ── Ultrasound ────────────────────────────────────────────────────────────
    if modality in ("ultrasound", "us"):
        return _native_model(
            f"ultrasound_{H}x{W}", (H, W),
            "pwm_core.physics.ultrasound.ultrasound_operator.UltrasoundOperator",
            {"modality": "ultrasound"},
            nz=H, nx=W, n_elements=n_elements, n_samples=n_samples,
            speed_of_sound=speed_of_sound, element_pitch=element_pitch, fs=fs,
        )

    # ── Photoacoustic ─────────────────────────────────────────────────────────
    if modality in ("photoacoustic", "pa", "pat"):
        return _native_model(
            f"photoacoustic_{H}x{W}", (H, W),
            "pwm_core.physics.photoacoustic.pa_operator.PAOperator",
            {"modality": "photoacoustic"},
            ny=H, nx=W, n_transducers=n_transducers,
            speed_of_sound=pa_speed_of_sound,
        )

    known = ("cassi, mri, ct, lensless, holography, ptychography, "
             "fluorescence, lightsheet, ultrasound, photoacoustic")
    raise ValueError(f"unknown modality {modality!r}; known: [{known}]")


def from_spec_fields(fields: Dict[str, Any], *,
                     mask: Optional[np.ndarray] = None) -> ForwardModel:
    """Build a ForwardModel from digital-twin spec fields (six_tuple/protocol).

    Recognises the same field layout produced by the optics pwm_bridge.
    """
    modality = (fields.get("spec_type") or fields.get("modality") or "").lower()
    omega = (fields.get("six_tuple") or {}).get("omega", {})
    H = int(omega.get("H", 0)) or int(fields.get("H", 0))
    W = int(omega.get("W", 0)) or int(fields.get("W", 0))
    N_bands = int(omega.get("N_bands", 1) or 1)
    if modality == "cassi":
        pf = fields.get("protocol_fields", {})
        a1 = float(pf.get("disp_a1_nominal", 1.0))
        disp = {"dispersion_model": "poly",
                "disp_poly_x": [0.0, a1], "disp_poly_y": [0.0, 0.0]}
        return from_modality("cassi", H=H, W=W, N_bands=N_bands,
                             mask=mask, dispersion=disp)
    # Delegate to from_modality for all other recognised modalities.
    try:
        return from_modality(modality, H=H, W=W)
    except ValueError:
        raise ValueError(f"from_spec_fields: unsupported modality {modality!r}")
