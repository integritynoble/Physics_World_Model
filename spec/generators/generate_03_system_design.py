#!/usr/bin/env python3
"""Usage 3: System Design — one minimal spec per modality. ~15 lines."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import AB_DIR, SPEC_DIR, CALIB_FN, SYSTEM_DAGS, extract_modality_info, get_input

OUT_DIR  = SPEC_DIR / '03_system_design'
PAPER_OUT = SPEC_DIR.parent / 'papers' / 'system_design' / 'outputs'

PAPER_MAP = {
    'ct':       'ct_forward_v1_iter1.md',
    'lensless': 'lensless_forward_v1_iter1.md',
    'sim':      'sim_forward_v1_iter1.md',
}


def extract_dag(fwd_path):
    """Pull the first code block from paper output."""
    txt = fwd_path.read_text(errors='replace')
    m = re.search(r'```\n(.*?)```', txt, re.DOTALL)
    return m.group(1).strip() if m else ''


def make_spec(mod_id, disp_name, solvers, dag, calib):
    inp     = get_input(mod_id)
    gcs     = f'gs://pwm-benchmark-datasets/datasets/Benchmark/{mod_id}/public/'
    n_total = sum(1 for v in solvers.values() if isinstance(v, dict))
    best    = next((k for k, v in solvers.items()
                    if isinstance(v, dict) and not v.get('gpu')), 'traditional_cpu')
    param   = calib.get('param', 'operator model error')
    rng     = calib.get('range', 'modality-dependent')
    calib_call  = calib.get('call', '').split('\n')[-1]
    calib_apply = calib.get('apply', 'calib_cfg = {}')
    calib_imp   = calib.get('import', '').split('\n')[0]

    paper_ref = ''
    if mod_id in PAPER_MAP:
        paper_ref = f'\n**Paper**: `papers/system_design/outputs/{PAPER_MAP[mod_id]}`'

    return f"""\
# {disp_name} — System Design

```
{dag}
```

**Mismatch**: {param} `{rng}`
**Input**: {inp}  **Algorithms**: {n_total} — see `spec/{mod_id}.md`
**Benchmark**: `{gcs}`{paper_ref}

```python
from algorithm_base.{mod_id}.solvers import run_solver
{calib_imp}
{calib_call}
{calib_apply}
x = run_solver('{best}', y, cfg=calib_cfg)
```
"""


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for mod_dir in sorted(d for d in AB_DIR.iterdir() if d.is_dir() and d.name != 'shared'):
        sp = mod_dir / 'solvers.py'
        if not sp.exists(): continue
        mod_id, disp_name, _, solvers = extract_modality_info(sp)
        calib = CALIB_FN.get(mod_id, {
            'import': '', 'call': '', 'apply': 'calib_cfg = {}',
            'param': 'operator model error', 'range': 'modality-dependent',
        })
        # DAG: from paper if available, else SYSTEM_DAGS dict, else default
        dag = ''
        if mod_id in PAPER_MAP:
            p = PAPER_OUT / PAPER_MAP[mod_id]
            if p.exists(): dag = extract_dag(p)
        if not dag:
            dag = SYSTEM_DAGS.get(mod_id,
                f'[Source] → [Forward ({disp_name})] → [Detector] → y\n'
                f'              ↓\n'
                f'          [Mismatch]')
        (OUT_DIR / f'{mod_id}.md').write_text(make_spec(mod_id, disp_name, solvers, dag, calib))
        total += 1
    print(f'Usage 3: {total} specs → {OUT_DIR}')

if __name__ == '__main__': run()
