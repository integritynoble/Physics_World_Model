#!/usr/bin/env python3
"""Usage 1: Reconstruct — one minimal spec per (modality, algorithm). ~15 lines each."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import AB_DIR, SPEC_DIR, extract_modality_info, extract_psnr, get_input

OUT_DIR = SPEC_DIR / '01_reconstruct'


def make_spec(mod_id, disp_name, solver_key, solver):
    name   = solver.get('name', solver_key)
    ref    = solver.get('reference', '')
    gpu    = bool(solver.get('gpu', False))
    cfg    = solver.get('cfg_override') or {}
    psnr   = extract_psnr(ref)
    device = 'GPU' if gpu else 'CPU'
    inp    = get_input(mod_id)
    gcs    = f'gs://pwm-benchmark-datasets/datasets/Benchmark/{mod_id}/public/'

    meta = f'**{device}**'
    if psnr: meta += f'  **PSNR**: {psnr}'
    if ref:  meta += f'  *{ref}*'

    cfg_line = f'cfg = {repr(cfg)}\n' if cfg else ''

    return f"""\
# {disp_name} — {name}

{meta}
**Input**: {inp}
**Benchmark**: `{gcs}`

```python
from algorithm_base.{mod_id}.solvers import run_solver
{cfg_line}x = run_solver('{solver_key}', y{', cfg=cfg' if cfg else ''})
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
"""


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for mod_dir in sorted(d for d in AB_DIR.iterdir() if d.is_dir() and d.name != 'shared'):
        sp = mod_dir / 'solvers.py'
        if not sp.exists(): continue
        mod_id, disp_name, _, solvers = extract_modality_info(sp)
        if not solvers: continue
        out = OUT_DIR / mod_id
        out.mkdir(exist_ok=True)
        for key, v in solvers.items():
            if not isinstance(v, dict): continue
            (out / f'{key}.md').write_text(make_spec(mod_id, disp_name, key, v))
            total += 1
    print(f'Usage 1: {total} specs → {OUT_DIR}')

if __name__ == '__main__': run()
