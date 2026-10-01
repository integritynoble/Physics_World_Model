"""The external validator for low-dose CT: the one that decides acceptance into the library.

Independent of the solver and of the runtime's judge: this file imports neither, and is the only place that reads the
protected answer key for a problem. The metric definitions are the field's published ones (the same as the runtime's
judge states them), implemented again here so that a defect in one implementation cannot pass itself:

- PSNR against the full-dose reference, with the peak taken as the reference's range; it must beat the untouched
  low-dose slice (a restoration that makes the image worse fails);
- lesion CNR: lesion-core mean minus background mean, over the background's standard deviation, where the background is
  soft tissue (-200..300 HU on the reference) in an annulus 2r..5r around the lesion; it must reach 3 (Rose);
- contrast retained: the core contrast over the inserted amplitude; at least half must survive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

NAME = "validator-ldct-1"


def _psnr(a: np.ndarray, ref: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - ref) ** 2))
    peak = float(max(ref.max() - ref.min(), 1e-9))
    return 10.0 * np.log10(peak ** 2 / max(mse, 1e-12))


def metrics(key_dir: Path, rec: np.ndarray) -> Dict[str, float]:
    key_dir = Path(key_dir)
    ref = np.load(key_dir / "full_dose.npy").astype(np.float64)
    low = np.load(key_dir / "low_dose.npy").astype(np.float64)
    mask = np.load(key_dir / "lesion_mask.npy").astype(bool)
    hu, r_les = (float(x) for x in np.load(key_dir / "lesion_amplitude.npy"))
    rec = np.asarray(rec, dtype=np.float64)
    if rec.shape != ref.shape or not np.isfinite(rec).all():
        raise ValueError(f"a reconstruction must be finite with shape {ref.shape}")
    ys, xs = np.nonzero(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    yy, xx = np.ogrid[:ref.shape[0], :ref.shape[1]]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    r = int(np.sqrt(mask.sum() / np.pi)) + 1
    annulus = (d2 > (2 * r) ** 2) & (d2 <= (5 * r) ** 2)
    background = annulus & (ref > -200) & (ref < 300)
    if background.sum() < 50:
        background = annulus
    core = d2 <= max(1.0, (r_les * 0.6) ** 2)
    contrast = float(rec[core].mean() - rec[background].mean())
    noise = float(rec[background].std())
    return {"psnr": _psnr(rec, ref), "psnr_before": _psnr(low, ref),
            "lesion_cnr": abs(contrast) / max(noise, 1e-9),
            "lesion_contrast_retained": abs(contrast) / max(abs(hu), 1e-9), "noise_hu": noise}


def verdict(m: Dict[str, float]) -> Dict[str, Any]:
    reasons = []
    if m["psnr"] <= m["psnr_before"]:
        reasons.append("PSNR not above the untouched low-dose slice")
    if m["lesion_cnr"] < 3.0:
        reasons.append("lesion CNR below 3 (Rose)")
    if m["lesion_contrast_retained"] < 0.5:
        reasons.append("less than half of the lesion contrast retained")
    return {"passed": not reasons, "reasons": reasons or ["all three criteria met"]}


def validate(key_dir: Path, rec: np.ndarray) -> Dict[str, Any]:
    m = metrics(key_dir, rec)
    v = verdict(m)
    # The objective the library ranks by: detectability, counted only when the guardrails hold.
    v.update(validator=NAME, metrics=m, objective=m["lesion_cnr"] if v["passed"] else 0.0)
    return v
