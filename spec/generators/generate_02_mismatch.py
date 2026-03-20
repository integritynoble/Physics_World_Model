#!/usr/bin/env python3
"""Usage 2: Mismatch Correction + Reconstruct — one spec per modality.

Output: spec/02_mismatch/{modality}.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    AB_DIR, SPEC_DIR, MISMATCH, BEST_CPU,
    extract_modality_info, extract_psnr, get_input, viz_code
)

OUT_DIR = SPEC_DIR / '02_mismatch'

# ── Default mismatch template for modalities not in MISMATCH dict ─────────
def default_mismatch(mod_id: str, disp_name: str) -> dict:
    return {
        'param': 'operator model error',
        'range': 'modality-dependent',
        'typical': 'calibration dependent',
        'method': 'grid search on reconstruction quality',
        'cfg_key': 'mismatch_param',
    }


def find_best_cpu(solvers: dict, mod_id: str) -> tuple[str, str]:
    """Return (key, name) of best CPU solver."""
    preferred = BEST_CPU.get(mod_id)
    if preferred and preferred in solvers:
        return preferred, solvers[preferred].get('name', preferred)
    # Fall back: first non-GPU solver with PSNR, or just first CPU
    best_key, best_name, best_psnr = 'traditional_cpu', 'baseline', 0.0
    for k, v in solvers.items():
        if not isinstance(v, dict) or v.get('gpu', False):
            continue
        ref = v.get('reference', '')
        import re
        m = re.search(r'(\d+\.?\d*)\s*dB', ref)
        psnr = float(m.group(1)) if m else 0.0
        if psnr > best_psnr:
            best_psnr = psnr
            best_key = k
            best_name = v.get('name', k)
    return best_key, best_name


def make_spec(mod_id: str, disp_name: str, dataset_line: str,
              solvers: dict) -> str:
    mm = MISMATCH.get(mod_id, default_mismatch(mod_id, disp_name))
    inp = get_input(mod_id)
    gcs = f'gs://pwm-benchmark-datasets/datasets/Benchmark/{mod_id}/public/'
    best_key, best_name = find_best_cpu(solvers, mod_id)
    viz = viz_code(mod_id, f'corrected')

    # Count solvers
    n_cpu = sum(1 for v in solvers.values()
                if isinstance(v, dict) and not v.get('gpu', False))
    n_gpu = sum(1 for v in solvers.values()
                if isinstance(v, dict) and v.get('gpu', False))

    spec = f"""\
# {disp_name} — Mismatch Correction + Reconstruct

**Input**: {inp}
**Benchmark**: `{gcs}`
**Algorithms**: {n_cpu} CPU, {n_gpu} GPU — reconstruct with `run_solver(key, y)`

## Mismatch

| Parameter | Range | Typical Error | Correction Method |
|-----------|-------|---------------|-------------------|
| {mm['param']} | `{mm['range']}` | {mm['typical']} | {mm['method']} |

## User Parameters

```python
# Set known values or leave None to auto-calibrate
mismatch_cfg = {{
    '{mm['cfg_key']}': None,    # float if known; None = auto-estimate
    'search_range': '{mm["range"]}',
    'search_steps': 20,
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
from algorithm_base.{mod_id}.solvers import run_solver, list_solvers
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Reconstruct WITHOUT mismatch correction (baseline)
x_wrong = run_solver('{best_key}', y)

# 2. Reconstruct WITH mismatch correction
x_corrected = run_solver('{best_key}', y, cfg=mismatch_cfg)

# 3. Evaluate
if x_true is not None:
    print(f"No correction  PSNR {{compute_psnr(x_true, x_wrong):.2f}} dB  "
          f"SSIM {{compute_ssim(x_true, x_wrong):.4f}}")
    print(f"Corrected      PSNR {{compute_psnr(x_true, x_corrected):.2f}} dB  "
          f"SSIM {{compute_ssim(x_true, x_corrected):.4f}}")

# 4. Use any other solver
# list_solvers()
# x = run_solver('your_key', y, cfg=mismatch_cfg)

x = x_corrected  # for visualization below
{viz}
```

## Reference

- `packages/pwm_core/contrib/mismatch_db.yaml` — mismatch parameter ranges
- `papers/system_design/` — system design with mismatch analysis
"""
    return spec


def run():
    modality_dirs = sorted(d for d in AB_DIR.iterdir()
                           if d.is_dir() and d.name != 'shared')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for mod_dir in modality_dirs:
        solvers_py = mod_dir / 'solvers.py'
        if not solvers_py.exists():
            continue
        mod_id, disp_name, dataset_line, solvers = extract_modality_info(solvers_py)

        spec = make_spec(mod_id, disp_name, dataset_line, solvers)
        (OUT_DIR / f'{mod_id}.md').write_text(spec, encoding='utf-8')
        total += 1

    print(f'Usage 2: generated {total} mismatch specs in {OUT_DIR}')


if __name__ == '__main__':
    run()
