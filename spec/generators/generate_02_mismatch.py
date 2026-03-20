#!/usr/bin/env python3
"""Usage 2: Mismatch Correction + Reconstruct — one spec per (modality, algorithm).

Each entry = reconstruction algorithm + gradient mismatch correction, matching
the benchmark leaderboard format: "{Algorithm} + gradient".

Output: spec/02_mismatch/{modality}/{solver_key}.md
"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    AB_DIR, SPEC_DIR, MISMATCH, VOLUMETRIC, HYPERSPECTRAL,
    extract_modality_info, extract_psnr, extract_ssim, get_input, viz_code
)

OUT_DIR = SPEC_DIR / '02_mismatch'

# ── Calibration function per modality (from pwm_core.mismatch.operators) ──
CALIB_FN = {
    'ct': {
        'fn': 'ct_calibrate_cor',
        'import': 'from pwm_core.mismatch.operators import ct_calibrate_cor',
        'call': 'cor_offset = ct_calibrate_cor(y, shift_range=5)',
        'apply': 'cfg = {"cor_offset": float(cor_offset)}',
        'param': 'CoR offset',
        'range': '[-5, +5] px',
        'typical': '±1 px',
        'grad_desc': 'Differentiable Radon transform; gradient descent refines CoR offset',
    },
    'mri': {
        'fn': 'mri_estimate_sensitivities_acs',
        'import': 'from pwm_core.mismatch.operators import mri_estimate_sensitivities_acs',
        'call': 'sens_maps = mri_estimate_sensitivities_acs(y, acs_lines=24)',
        'apply': 'cfg = {"sens_maps": sens_maps}',
        'param': 'Coil sensitivity maps (B1 error)',
        'range': '[0.9, 1.1] gain per coil',
        'typical': '±5% per coil',
        'grad_desc': 'ESPIRiT auto-calibration from ACS center lines',
    },
    'cassi': {
        'fn': 'cassi_calibrate_step',
        'import': 'from pwm_core.mismatch.operators import cassi_calibrate_step',
        'call': 'disp_step = cassi_calibrate_step(y, step_range=[1, 5], n_steps=20)',
        'apply': 'cfg = {"disp_step": int(round(disp_step))}',
        'param': 'Dispersion step (px)',
        'range': '[1, 5] px',
        'typical': '±0.5 px drift',
        'grad_desc': 'Sparsity-based grid search; gradient refines sub-pixel step',
    },
    'cacti': {
        'fn': 'cacti_calibrate_timing',
        'import': 'from pwm_core.mismatch.operators import cacti_calibrate_timing',
        'call': 'timing_offset = cacti_calibrate_timing(y)',
        'apply': 'cfg = {"timing_offset": float(timing_offset)}',
        'param': 'Exposure timing offset (frames)',
        'range': '[-2, +2] frames',
        'typical': '±0.5 frame',
        'grad_desc': 'Cross-correlation on coded frames; gradient refines sub-frame alignment',
    },
    'spc': {
        'fn': 'spc_calibrate_gain_bias',
        'import': 'from pwm_core.mismatch.operators import spc_calibrate_gain_bias',
        'call': 'gain, bias = spc_calibrate_gain_bias(y)',
        'apply': 'cfg = {"gain": float(gain), "bias": float(bias)}',
        'param': 'Sensor gain and dark bias',
        'range': 'gain [0.8, 1.2], bias [0, 50]',
        'typical': '±10% gain, ±5 counts bias',
        'grad_desc': 'NLL-based gain/bias estimation; gradient refines via photon statistics',
    },
    'lensless': {
        'fn': 'lensless_calibrate_shift',
        'import': 'from pwm_core.mismatch.operators import lensless_calibrate_shift\nfrom scipy.ndimage import gaussian_filter',
        'call': ('psf = gaussian_filter(psf_nominal, sigma=5); psf /= psf.sum()\n'
                 'shift = lensless_calibrate_shift(y, psf, shift_range=5)'),
        'apply': 'cfg = {"psf_shift_y": float(shift[0]), "psf_shift_x": float(shift[1])}',
        'param': 'PSF shift (px)',
        'range': '[-5, +5] px (x and y)',
        'typical': '±1 px thermal drift',
        'grad_desc': 'Gradient w.r.t. PSF shift; minimizes ||Hx - y|| with shifted PSF',
    },
    'cbct': {
        'fn': 'cbct_calibrate_offset',
        'import': 'from pwm_core.mismatch.operators import cbct_calibrate_offset',
        'call': 'geo_offset = cbct_calibrate_offset(y)',
        'apply': 'cfg = {"geo_offset": float(geo_offset)}',
        'param': 'Source-detector geometry offset (mm)',
        'range': '[-10, +10] mm',
        'typical': '±2 mm per session',
        'grad_desc': 'Phantom-based calibration; gradient descent on geometric projection error',
    },
    'ptychography': {
        'fn': 'ptycho_calibrate_offset',
        'import': 'from pwm_core.mismatch.operators import ptycho_calibrate_offset',
        'call': 'pos_correction = ptycho_calibrate_offset(patterns, positions_nominal)',
        'apply': 'cfg = {"position_correction": pos_correction}',
        'param': 'Probe position error (px)',
        'range': '[-3, +3] px',
        'typical': '±1 px',
        'grad_desc': 'Gradient-based position refinement (annealing); ePIE with position update',
    },
    'cryo_em': {
        'fn': 'cryoem_calibrate_defocus',
        'import': 'from pwm_core.mismatch.operators import cryoem_calibrate_defocus',
        'call': 'defocus = cryoem_calibrate_defocus(micrograph)',
        'apply': 'cfg = {"defocus_um": float(defocus)}',
        'param': 'CTF defocus (μm)',
        'range': '[0.5, 4.0] μm',
        'typical': '±0.2 μm',
        'grad_desc': 'CTF fitting in frequency domain; gradient refines defocus and astigmatism',
    },
    'ultrasound': {
        'fn': 'ultrasound_calibrate_sos',
        'import': 'from pwm_core.mismatch.operators import ultrasound_calibrate_sos',
        'call': 'sos = ultrasound_calibrate_sos(rf_data, sos_range=[1400, 1600])',
        'apply': 'cfg = {"c0": float(sos)}',
        'param': 'Speed of sound (m/s)',
        'range': '[1400, 1600] m/s',
        'typical': '±30 m/s tissue variation',
        'grad_desc': 'Coherence-based SoS estimation; gradient maximizes beamformed coherence',
    },
    'industrial_ct': {
        'fn': 'ct_calibrate_cor',
        'import': 'from pwm_core.mismatch.operators import ct_calibrate_cor',
        'call': 'cor_offset = ct_calibrate_cor(y, shift_range=10)',
        'apply': 'cfg = {"cor_offset": float(cor_offset)}',
        'param': 'CoR offset (px)',
        'range': '[-10, +10] px',
        'typical': '±2 px',
        'grad_desc': 'Sinogram symmetry; gradient descent on projection consistency metric',
    },
    'widefield': {
        'fn': 'fluorescence_calibrate_sigma',
        'import': 'from pwm_core.mismatch.operators import fluorescence_calibrate_sigma',
        'call': 'psf_sigma = fluorescence_calibrate_sigma(y, sigma_range=[0.5, 3.0])',
        'apply': 'cfg = {"psf_sigma": float(psf_sigma)}',
        'param': 'PSF sigma (px)',
        'range': '[0.5, 3.0] px',
        'typical': '±0.3 px',
        'grad_desc': 'Sharpness-based grid search; gradient refines PSF width via Richardson-Lucy',
    },
}

# ── Default calibration for modalities without dedicated function ──────────
def default_calib(mod_id: str, disp_name: str) -> dict:
    return {
        'fn': 'grid_search',
        'import': '',
        'call': '# Use grid search over mismatch parameter',
        'apply': 'cfg = {"mismatch_param": None}  # None = auto-estimate',
        'param': 'operator model parameter',
        'range': 'modality-dependent',
        'typical': 'calibration-dependent',
        'grad_desc': 'Grid search on reconstruction quality metric; gradient refines optimal parameter',
    }


def make_spec(mod_id: str, disp_name: str, dataset_line: str,
              solver_key: str, solver: dict,
              calib: dict) -> str:
    name    = solver.get('name', solver_key)
    ref     = solver.get('reference', '')
    gpu     = bool(solver.get('gpu', False))
    cfg_ov  = solver.get('cfg_override', {}) or {}
    psnr    = extract_psnr(ref)
    ssim    = extract_ssim(ref)
    device  = 'GPU' if gpu else 'CPU'
    inp     = get_input(mod_id)
    gcs     = f'gs://pwm-benchmark-datasets/datasets/Benchmark/{mod_id}/public/'
    viz     = viz_code(mod_id, f'{solver_key}_corrected')

    metrics = []
    if psnr: metrics.append(f'**PSNR**: {psnr}')
    if ssim: metrics.append(f'**SSIM**: {ssim}')
    metrics_line = '  '.join(metrics) if metrics else ''

    # Reconstruction cfg (from solver cfg_override merged with mismatch cfg)
    recon_cfg = repr(cfg_ov) if cfg_ov else '{}'

    calib_import = calib['import']
    calib_call   = calib['call']
    calib_apply  = calib['apply']

    spec = f"""\
# {disp_name} — {name} + Gradient Mismatch Correction

**Device**: {device}  **Input**: {inp}
{'**Reference**: ' + ref if ref else ''}
{metrics_line}

## Mismatch

| Parameter | Range | Typical | Gradient Correction |
|-----------|-------|---------|---------------------|
| {calib['param']} | `{calib['range']}` | {calib['typical']} | {calib['grad_desc']} |

## Correction Parameters

```python
mismatch_cfg = {{
    '{calib["param"].split("(")[0].strip().lower().replace(" ", "_").replace("-", "_")}': None,  # None = auto-calibrate
    'search_range': '{calib["range"]}',
    'grad_steps': 50,
    'grad_lr': 0.01,
}}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('{mod_id}_public.h5', 'r') as f:   # GCS: {gcs}
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # {inp}
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.{mod_id}.solvers import run_solver
{calib_import}
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Baseline — reconstruct WITHOUT mismatch correction
x_wrong = run_solver('{solver_key}', y, cfg={recon_cfg})

# 2. Calibrate mismatch parameter
{calib_call}

# 3. Reconstruct WITH correction
{calib_apply}
cfg.update({recon_cfg})
x_corrected = run_solver('{solver_key}', y, cfg=cfg)

# 4. Evaluate
if x_true is not None:
    print(f"No correction  PSNR {{compute_psnr(x_true, x_wrong):.2f}} dB  "
          f"SSIM {{compute_ssim(x_true, x_wrong):.4f}}")
    print(f"Corrected      PSNR {{compute_psnr(x_true, x_corrected):.2f}} dB  "
          f"SSIM {{compute_ssim(x_true, x_corrected):.4f}}")

x = x_corrected
{viz}
```

## Benchmark

`{disp_name} — {name} + gradient` corresponds to the leaderboard entry at
`https://pwm.platformai.org/benchmark/{mod_id}` (mismatch correction tier).
"""
    return spec


def run():
    modality_dirs = sorted(d for d in AB_DIR.iterdir()
                           if d.is_dir() and d.name != 'shared')

    # Remove old flat .md files first
    for f in OUT_DIR.glob('*.md'):
        f.unlink()

    total = 0
    for mod_dir in modality_dirs:
        solvers_py = mod_dir / 'solvers.py'
        if not solvers_py.exists():
            continue
        mod_id, disp_name, dataset_line, solvers = extract_modality_info(solvers_py)
        if not solvers:
            continue

        calib = CALIB_FN.get(mod_id, default_calib(mod_id, disp_name))
        out_mod = OUT_DIR / mod_id
        out_mod.mkdir(parents=True, exist_ok=True)

        for solver_key, solver in solvers.items():
            if not isinstance(solver, dict):
                continue
            spec = make_spec(mod_id, disp_name, dataset_line,
                             solver_key, solver, calib)
            (out_mod / f'{solver_key}.md').write_text(spec, encoding='utf-8')
            total += 1

    print(f'Usage 2: generated {total} per-algorithm mismatch specs in {OUT_DIR}')


if __name__ == '__main__':
    run()
