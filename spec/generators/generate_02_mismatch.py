#!/usr/bin/env python3
"""Usage 2: Mismatch + Reconstruct — one minimal spec per (modality, algorithm). ~15 lines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import AB_DIR, SPEC_DIR, CALIB_FN, extract_modality_info, extract_psnr, get_input

OUT_DIR = SPEC_DIR / '02_mismatch'


def make_spec(mod_id, disp_name, solver_key, solver, calib):
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

    calib_import = calib['import'].split('\n')[0]  # first line only
    calib_call   = calib['call'].split('\n')[-1]   # last line (the actual call)
    calib_apply  = calib['apply']
    param        = calib['param']
    rng          = calib['range']

    cfg_str = f', cfg={{**calib_cfg, **{repr(cfg)}}}' if cfg else ', cfg=calib_cfg'

    return f"""\
# {disp_name} — {name} + Gradient

{meta}  **Mismatch**: {param} `{rng}`
**Input**: {inp}
**Benchmark**: `{gcs}`

```python
from algorithm_base.{mod_id}.solvers import run_solver
{calib_import}

x_wrong = run_solver('{solver_key}', y)           # no correction
{calib_call}
{calib_apply}
x = run_solver('{solver_key}', y{cfg_str})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
"""


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUT_DIR.glob('*.md'): f.unlink()  # remove old flat files
    total = 0
    for mod_dir in sorted(d for d in AB_DIR.iterdir() if d.is_dir() and d.name != 'shared'):
        sp = mod_dir / 'solvers.py'
        if not sp.exists(): continue
        mod_id, disp_name, _, solvers = extract_modality_info(sp)
        if not solvers: continue
        calib = CALIB_FN.get(mod_id, {
            'import': '', 'call': '# auto-calibrate mismatch parameter',
            'apply': 'calib_cfg = {"mismatch_param": None}',
            'param': 'operator model error', 'range': 'modality-dependent',
        })
        out = OUT_DIR / mod_id
        out.mkdir(exist_ok=True)
        for key, v in solvers.items():
            if not isinstance(v, dict): continue
            (out / f'{key}.md').write_text(make_spec(mod_id, disp_name, key, v, calib))
            total += 1
    print(f'Usage 2: {total} specs → {OUT_DIR}')

if __name__ == '__main__': run()
