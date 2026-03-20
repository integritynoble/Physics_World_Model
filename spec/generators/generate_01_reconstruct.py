#!/usr/bin/env python3
"""Usage 1: Reconstruct — one spec per (modality, algorithm).

Output: spec/01_reconstruct/{modality}/{solver_key}.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    AB_DIR, SPEC_DIR, INPUT_FORMATS, VOLUMETRIC, HYPERSPECTRAL,
    extract_modality_info, extract_psnr, extract_ssim, get_input, viz_code
)

OUT_DIR = SPEC_DIR / '01_reconstruct'


def params_table(cfg: dict) -> str:
    if not cfg:
        return ''
    rows = ['| Parameter | Default |', '|-----------|---------|']
    for k, v in cfg.items():
        rows.append(f'| `{k}` | `{v}` |')
    return '\n'.join(rows)


def make_spec(mod_id: str, disp_name: str, dataset_line: str,
              solver_key: str, solver: dict) -> str:
    name     = solver.get('name', solver_key)
    ref      = solver.get('reference', '')
    gpu      = bool(solver.get('gpu', False))
    cfg_ov   = solver.get('cfg_override', {}) or {}
    psnr     = extract_psnr(ref)
    ssim     = extract_ssim(ref)
    device   = 'GPU' if gpu else 'CPU'
    inp      = get_input(mod_id)
    gcs      = f'gs://pwm-benchmark-datasets/datasets/Benchmark/{mod_id}/public/'

    # Metrics line
    metrics = []
    if psnr: metrics.append(f'**PSNR**: {psnr}')
    if ssim: metrics.append(f'**SSIM**: {ssim}')
    metrics_line = '  '.join(metrics) if metrics else '*No reference metric in registry*'

    # Params table
    ptable = params_table(cfg_ov)

    # cfg string for code
    cfg_str = repr(cfg_ov) if cfg_ov else '{}'

    # visualization
    viz = viz_code(mod_id, solver_key)

    ref_line = f'*{ref}*' if ref else ''

    spec = f"""\
# {disp_name} — {name}

**Device**: {device}  **Input**: {inp}
{'**Dataset**: ' + dataset_line if dataset_line else ''}
{metrics_line}

## System

{dataset_line or f'{disp_name} reconstruction problem.'}

## Algorithm Parameters

{ptable if ptable else '*Uses default configuration.*'}

## Measurement

```python
# Option A — PWM benchmark (ground truth available → PSNR/SSIM)
import h5py
with h5py.File('{mod_id}_public.h5', 'r') as f:   # download from GCS below
    y, x_true = f['y'][0], f['x_true'][0]
# GCS: {gcs}

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # {inp}
x_true = None   # provide your ground truth for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.{mod_id}.solvers import run_solver
from pwm_core.utils.metrics import compute_psnr, compute_ssim

cfg = {cfg_str}
x   = run_solver('{solver_key}', y, cfg=cfg)

if x_true is not None:
    print(f"PSNR {{compute_psnr(x_true, x):.2f}} dB  SSIM {{compute_ssim(x_true, x):.4f}}")

{viz}
```

## Reference

{ref_line}
"""
    return spec


def run():
    modality_dirs = sorted(d for d in AB_DIR.iterdir()
                           if d.is_dir() and d.name != 'shared')
    total = 0
    for mod_dir in modality_dirs:
        solvers_py = mod_dir / 'solvers.py'
        if not solvers_py.exists():
            continue
        mod_id, disp_name, dataset_line, solvers = extract_modality_info(solvers_py)
        if not solvers:
            continue

        out_mod = OUT_DIR / mod_id
        out_mod.mkdir(parents=True, exist_ok=True)

        for solver_key, solver in solvers.items():
            if not isinstance(solver, dict):
                continue
            spec = make_spec(mod_id, disp_name, dataset_line,
                             solver_key, solver)
            (out_mod / f'{solver_key}.md').write_text(spec, encoding='utf-8')
            total += 1

    print(f'Usage 1: generated {total} per-algorithm specs in {OUT_DIR}')


if __name__ == '__main__':
    run()
