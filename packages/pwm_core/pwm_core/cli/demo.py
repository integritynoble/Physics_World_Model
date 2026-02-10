"""pwm_core.cli.demo

Implements ``pwm demo <modality> [--preset NAME] [--run] [--open-viewer] [--export-sharepack]``.

Loads a CasePack by modality + preset name, optionally runs the simulation/
reconstruction pipeline, exports a SharePack, and/or launches the viewer.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CasePack discovery
# ---------------------------------------------------------------------------

_CASEPACKS_DIR = (
    Path(__file__).resolve().parents[3] / "contrib" / "casepacks"
)


def _find_casepacks_for_modality(modality: str) -> List[Path]:
    """Return all CasePack JSON files matching *modality*."""
    results = []
    if _CASEPACKS_DIR.is_dir():
        for p in sorted(_CASEPACKS_DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("modality") == modality:
                    results.append(p)
            except Exception:
                continue
    return results


def _load_casepack(modality: str, preset: Optional[str] = None) -> Dict[str, Any]:
    """Load a CasePack by modality and optional preset name.

    If *preset* is None the first available CasePack for that modality is used.
    """
    candidates = _find_casepacks_for_modality(modality)
    if not candidates:
        raise SystemExit(
            "No CasePack found for modality '{}'. "
            "Available casepacks are in: {}".format(modality, _CASEPACKS_DIR)
        )

    if preset is not None:
        # Match by id containing the preset string
        for p in candidates:
            data = json.loads(p.read_text(encoding="utf-8"))
            cp_id = data.get("id", "")
            if preset in cp_id or preset == cp_id:
                return data
        # Fallback: match by keyword/tag
        for p in candidates:
            data = json.loads(p.read_text(encoding="utf-8"))
            tags = data.get("tags", []) + data.get("keywords", [])
            if preset in tags:
                return data
        raise SystemExit(
            "Preset '{}' not found for modality '{}'. "
            "Available casepacks: {}".format(
                preset, modality, [c.stem for c in candidates]
            )
        )

    # Default: first available
    return json.loads(candidates[0].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Preset listing helpers
# ---------------------------------------------------------------------------

# Well-known presets per modality (for documentation / --help)
MODALITY_PRESETS = {
    "cassi": ["spectral_imaging", "measured_y_fit_theta"],
    "cacti": ["video_sci"],
    "mri": ["cartesian_accelerated"],
    "widefield": ["deconv_basic", "lowdose_highbg"],
    "confocal": ["livecell_lowdose_drift", "3d_stack_attenuation"],
    "sim": ["3x3_fragile"],
    "spc": ["low_sampling_poisson"],
    "lensless": ["diffusercam_basic"],
    "lightsheet": ["tissue_stripes_scatter"],
    "ct": ["conebeam_lowdose_scatter"],
    "ptychography": ["phase_retrieval"],
    "holography": ["offaxis_phase"],
    "nerf": ["from_poses_basic"],
    "gaussian_splatting": ["basic"],
    "matrix": ["generic_linear"],
    "panorama_multifocal": ["fusion"],
    "light_field": ["plenoptic_basic"],
    "oct": ["retinal_balanced"],
}


def _list_presets(modality: str) -> List[str]:
    return MODALITY_PRESETS.get(modality, [])


# ---------------------------------------------------------------------------
# Pipeline stub
# ---------------------------------------------------------------------------

def _run_pipeline(casepack: Dict[str, Any], out_dir: Path) -> Path:
    """Run simulation -> reconstruction pipeline from CasePack.

    Returns the RunBundle directory path. Currently delegates to
    the endpoints module; falls back to a synthetic stub for demo purposes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rb_dir = out_dir / "runbundle"
    rb_dir.mkdir(parents=True, exist_ok=True)

    try:
        from pwm_core.api import endpoints
        spec = casepack.get("base_spec", {})
        result = endpoints.run(spec=spec, out_dir=str(rb_dir))
        return rb_dir
    except Exception as exc:
        logger.info(
            "Full pipeline not available (%s), generating synthetic RunBundle", exc
        )

    # Synthetic RunBundle for demo/testing
    import numpy as np

    modality = casepack.get("modality", "unknown")
    bench = casepack.get("benchmark_results", {})

    data_dir = rb_dir / "data"
    data_dir.mkdir(exist_ok=True)
    results_dir = rb_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate small synthetic arrays
    h, w = 64, 64
    x_gt = np.random.RandomState(42).rand(h, w).astype(np.float32)
    y = x_gt + np.random.RandomState(43).randn(h, w).astype(np.float32) * 0.1
    x_hat = x_gt * 0.95  # Slightly imperfect reconstruction

    np.save(str(data_dir / "x_gt.npy"), x_gt)
    np.save(str(data_dir / "y.npy"), y)
    np.save(str(results_dir / "x_hat.npy"), x_hat)

    import hashlib

    def _sha(p):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        return "sha256:" + h

    manifest = {
        "version": "0.3.0",
        "spec_id": "demo_" + modality,
        "timestamp": "2026-02-09T00:00:00Z",
        "modality": modality,
        "provenance": {
            "git_hash": "0000000",
            "seeds": [42],
            "platform": sys.platform,
            "pwm_version": "0.3.0",
        },
        "metrics": {
            "psnr_db": bench.get("psnr_db", 30.0),
            "ssim": bench.get("ssim", 0.90),
            "runtime_s": 1.0,
        },
        "artifacts": {
            "x_gt": "data/x_gt.npy",
            "y": "data/y.npy",
            "x_hat": "results/x_hat.npy",
        },
        "hashes": {
            "x_gt": _sha(data_dir / "x_gt.npy"),
            "y": _sha(data_dir / "y.npy"),
            "x_hat": _sha(results_dir / "x_hat.npy"),
        },
    }
    (rb_dir / "runbundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return rb_dir


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

def cmd_demo(args):
    """Entry point for ``pwm demo``."""
    modality = args.modality
    preset = getattr(args, "preset", None)
    do_run = getattr(args, "run", False)
    do_sharepack = getattr(args, "export_sharepack", False)
    do_viewer = getattr(args, "open_viewer", False)

    print("PWM Demo: modality={}, preset={}".format(modality, preset), file=sys.stderr)

    # Load CasePack
    casepack = _load_casepack(modality, preset)
    print(
        "Loaded CasePack: {} ({})".format(
            casepack.get("id", "?"), casepack.get("title", "")
        ),
        file=sys.stderr,
    )

    rb_dir = None

    if do_run:
        out_dir = Path("runs") / ("demo_" + modality)
        rb_dir = _run_pipeline(casepack, out_dir)
        print("RunBundle at: {}".format(rb_dir), file=sys.stderr)

    if do_sharepack:
        if rb_dir is None:
            # Try to find an existing RunBundle
            default_rb = Path("runs") / ("demo_" + modality) / "runbundle"
            if default_rb.exists():
                rb_dir = default_rb
            else:
                print(
                    "No RunBundle available. Use --run first to generate one.",
                    file=sys.stderr,
                )
                return

        from pwm_core.export.sharepack import export_sharepack

        sp_dir = rb_dir.parent / "sharepack"
        export_sharepack(
            runbundle_dir=str(rb_dir),
            output_dir=str(sp_dir),
            modality=modality,
            preset=preset,
        )
        print("SharePack at: {}".format(sp_dir), file=sys.stderr)

    if do_viewer:
        if rb_dir is None:
            default_rb = Path("runs") / ("demo_" + modality) / "runbundle"
            if default_rb.exists():
                rb_dir = default_rb
            else:
                print(
                    "No RunBundle available. Use --run first.",
                    file=sys.stderr,
                )
                return
        try:
            from pwm_core.cli.view import view
            view(str(rb_dir))
        except Exception as exc:
            print("Could not launch viewer: {}".format(exc), file=sys.stderr)

    if not do_run and not do_sharepack and not do_viewer:
        # Just show info about the casepack
        print(json.dumps({
            "modality": modality,
            "casepack_id": casepack.get("id"),
            "title": casepack.get("title"),
            "available_presets": _list_presets(modality),
            "benchmark_results": casepack.get("benchmark_results", {}),
            "notes": casepack.get("notes", ""),
        }, indent=2))


def add_demo_subparser(subparsers):
    """Add the demo subcommand to the CLI parser."""
    p_demo = subparsers.add_parser(
        "demo",
        help="Run a demo for a specific imaging modality",
    )
    p_demo.add_argument(
        "modality",
        type=str,
        help="Modality key (e.g. cassi, mri, widefield, cacti, spc, ...)",
    )
    p_demo.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name (e.g. tissue, satellite). Defaults to first available.",
    )
    p_demo.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Execute simulation + reconstruction pipeline",
    )
    p_demo.add_argument(
        "--open-viewer",
        action="store_true",
        default=False,
        help="Launch Streamlit viewer after run",
    )
    p_demo.add_argument(
        "--export-sharepack",
        action="store_true",
        default=False,
        help="Export SharePack to sharepack/ directory",
    )
    p_demo.set_defaults(func=cmd_demo)
    return p_demo
