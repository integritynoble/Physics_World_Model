#!/usr/bin/env python3
"""Usage 3: Imaging System Design — one spec per modality (all 173).

Uses papers/system_design/outputs/*.md when available;
generates simplified system DAGs for all others.

Output: spec/03_system_design/{modality}.md
"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    AB_DIR, SPEC_DIR, MISMATCH, BEST_CPU,
    extract_modality_info, extract_psnr, get_input, viz_code
)

OUT_DIR = SPEC_DIR / '03_system_design'
PAPER_OUTPUTS = SPEC_DIR.parent / 'papers' / 'system_design' / 'outputs'

# ── Mapping from modality → paper output files ────────────────────────────
PAPER_MAP = {
    'ct':       ('ct_forward_v1_iter1.md', 'ct_reconstruction_v1_iter1.md'),
    'lensless': ('lensless_forward_v1_iter1.md', 'lensless_reconstruction_v1_iter1.md'),
    'sim':      ('sim_forward_v1_iter1.md', 'sim_reconstruction_v1_iter1.md'),
}

# ── Simplified system DAGs ────────────────────────────────────────────────
SYSTEM_DAGS = {
    'ct': """\
[X-ray Tube] → [Phantom] → [Parallel Geometry] → [Flat-Panel Detector] → [ADC] → y
      ↓                           ↓                        ↓
 [Beam hard.]            [CoR offset ±2 px]         [Poisson noise]""",

    'mri': """\
[RF Coils] → [Spin Excitation] → [k-space Sampling] → [ADC] → y
      ↓                ↓                  ↓
 [B1 error]      [Motion blur]     [Coil sensitivity error]""",

    'cassi': """\
[Broadband Source] → [Coded Aperture] → [Dispersive Prism] → [FPA Detector] → y
                           ↓                    ↓
                   [Mask misalign]       [Disp. step ±0.5 px]""",

    'lensless': """\
[LED] → [Diffuser] → [Object] → [CMOS Sensor] → y
            ↓                         ↓
     [PSF shift ±1 px]         [Poisson + readout noise]""",

    'cbct': """\
[X-ray Tube] → [Object] → [Cone Geometry] → [Flat Panel] → [ADC] → y
      ↓                         ↓                  ↓
 [Scatter]              [Geometry error]      [Lag artifact]""",

    'pet': """\
[Tracer Injection] → [Annihilation] → [Crystal Ring] → [Coincidence] → y
                                            ↓                ↓
                                     [Scatter fraction]  [Attenuation]""",

    'spect': """\
[Radiotracer] → [Collimator] → [NaI Crystal] → [PMT Array] → y
                     ↓                ↓
              [Septal penetration] [Scatter 25%]""",

    'oct': """\
[Broadband Source] → [Interferometer] → [Spectrometer] → [CCD] → y
                           ↓                   ↓
                    [Dispersion mismatch] [Phase noise]""",

    'ultrasound': """\
[Probe Array] → [Pulse-Echo] → [Tissue] → [RF Sampling] → y
                                   ↓              ↓
                           [Speed of sound]  [Reverberation]""",

    'sim': """\
[Laser] → [Diffraction Grating] → [Objective] → [Sample] → [Camera] → y
                   ↓                    ↓
           [Phase drift ±0.1 rad]  [Aberrations]""",

    'photoacoustic': """\
[Pulsed Laser] → [Tissue Absorption] → [Acoustic Emission] → [Transducer Array] → y
                         ↓                       ↓
                  [Fluence variation]      [Speed of sound error]""",

    'ptychography': """\
[Coherent Source] → [Probe] → [Sample] → [Far-field Detector] → y
        ↓                         ↓
 [Probe position error ±1 px]  [Partial coherence]""",

    'phase_retrieval': """\
[Coherent Source] → [Aperture] → [Propagation] → [Intensity Detector] → y
                                       ↓
                              [Distance calibration error]""",

    'fpm': """\
[LED Array] → [Sample] → [Objective] → [Camera] → y
      ↓                                    ↓
[LED position error ±0.5 mm]        [Poisson noise]""",
}

# ── Default DAG generator ─────────────────────────────────────────────────
def default_dag(mod_id: str, disp_name: str) -> str:
    return f"""\
[Source] → [Forward Model ({disp_name})] → [Detector] → y
      ↓                ↓
  [Noise]         [Mismatch]"""


def make_spec_from_paper(mod_id: str, disp_name: str,
                         fwd_path: Path, recon_path: Path,
                         solvers: dict) -> str:
    """Build spec using paper system_design output."""
    fwd_text = fwd_path.read_text(encoding='utf-8', errors='replace') if fwd_path.exists() else ''
    recon_text = recon_path.read_text(encoding='utf-8', errors='replace') if recon_path.exists() else ''

    # Extract DAG block from forward text
    dag = ''
    m = re.search(r'```\n(.*?)```', fwd_text, re.DOTALL)
    if m:
        dag = m.group(1).strip()

    # Extract element definitions (## Element or ### Element sections)
    elements = ''
    m = re.search(r'(## Element[^\n]*\n.*?)(?=##|\Z)', fwd_text, re.DOTALL)
    if m:
        elements = m.group(1).strip()[:800]  # cap at 800 chars

    # Extract mismatch summary
    mismatch_summary = ''
    for line in fwd_text.splitlines():
        if 'mismatch' in line.lower() and '[' in line:
            mismatch_summary += f'- {line.strip()}\n'
    mismatch_summary = mismatch_summary[:400]

    n_cpu = sum(1 for v in solvers.values()
                if isinstance(v, dict) and not v.get('gpu', False))
    n_gpu = sum(1 for v in solvers.values()
                if isinstance(v, dict) and v.get('gpu', False))
    inp = get_input(mod_id)
    gcs = f'gs://pwm-benchmark-datasets/datasets/Benchmark/{mod_id}/public/'
    best_key = BEST_CPU.get(mod_id, 'traditional_cpu')
    viz = viz_code(mod_id, 'recon')
    mm = MISMATCH.get(mod_id, {})
    mm_param = mm.get('param', 'operator model error') if mm else 'operator model error'
    mm_range = mm.get('range', 'modality-dependent') if mm else 'modality-dependent'

    spec = f"""\
# {disp_name} — System Design

> Source: `papers/system_design/outputs/{fwd_path.name}` + `{recon_path.name}`

## System DAG

```
{dag or default_dag(mod_id, disp_name)}
```

## Key Mismatch Sources

{mismatch_summary or f'- {mm_param}: `{mm_range}`'}

## Reconstruction

**Algorithms**: {n_cpu} CPU, {n_gpu} GPU
**Best CPU**: `{best_key}`
See `spec/{mod_id}.md` for full algorithm table.

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.{mod_id}.solvers import run_solver, list_solvers
import numpy as np, h5py
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# Load benchmark data (has ground truth)
with h5py.File('{mod_id}_public.h5', 'r') as f:   # GCS: {gcs}
    y, x_true = f['y'][0], f['x_true'][0]
# Or: y = np.load('your_measurement.npy').astype('float32'); x_true = None

# Simulate mismatch then correct
x = run_solver('{best_key}', y, cfg={{'{mm.get("cfg_key","mismatch_param") if mm else "mismatch_param"}': None}})

if x_true is not None:
    print(f"PSNR {{compute_psnr(x_true, x):.2f}} dB  SSIM {{compute_ssim(x_true, x):.4f}}")

{viz}
```

## Papers

- Forward model: `papers/system_design/outputs/{fwd_path.name}`
- Reconstruction: `papers/system_design/outputs/{recon_path.name}`
- Multi-agent: `python3 papers/system_design/main.py --modality {mod_id} --period forward`
"""
    return spec


def make_spec_generic(mod_id: str, disp_name: str, dataset_line: str,
                      solvers: dict) -> str:
    dag = SYSTEM_DAGS.get(mod_id, default_dag(mod_id, disp_name))
    inp = get_input(mod_id)
    gcs = f'gs://pwm-benchmark-datasets/datasets/Benchmark/{mod_id}/public/'
    best_key = BEST_CPU.get(mod_id, next(
        (k for k, v in solvers.items() if isinstance(v, dict) and not v.get('gpu', False)),
        'traditional_cpu'
    ))
    n_cpu = sum(1 for v in solvers.values()
                if isinstance(v, dict) and not v.get('gpu', False))
    n_gpu = sum(1 for v in solvers.values()
                if isinstance(v, dict) and v.get('gpu', False))
    mm = MISMATCH.get(mod_id, {})
    mm_param = mm.get('param', 'operator model error') if mm else 'operator model error'
    mm_range = mm.get('range', 'modality-dependent') if mm else 'modality-dependent'
    mm_method = mm.get('method', 'grid search on reconstruction quality') if mm else 'grid search'
    mm_key = mm.get('cfg_key', 'mismatch_param') if mm else 'mismatch_param'
    viz = viz_code(mod_id, 'recon')
    ds = dataset_line or disp_name

    spec = f"""\
# {disp_name} — System Design

## System DAG

```
{dag}
```

## System Elements

| Element | Type | Key Mismatch |
|---------|------|-------------|
| Source | illumination | intensity drift |
| Forward model | {disp_name} physics | {mm_param} |
| Detector | measurement | noise |

**Mismatch**: {mm_param} in range `{mm_range}`
**Correction**: {mm_method}

## Reconstruction

**Dataset**: {ds}
**Input**: {inp}
**Algorithms**: {n_cpu} CPU, {n_gpu} GPU — see `spec/{mod_id}.md`

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.{mod_id}.solvers import run_solver
import numpy as np, h5py
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# Load benchmark data (has ground truth)
with h5py.File('{mod_id}_public.h5', 'r') as f:   # GCS: {gcs}
    y, x_true = f['y'][0], f['x_true'][0]
# Or: y = np.load('your_measurement.npy').astype('float32'); x_true = None

# Forward model + mismatch correction + reconstruction
x = run_solver('{best_key}', y, cfg={{'{mm_key}': None}})  # None = auto-calibrate

if x_true is not None:
    print(f"PSNR {{compute_psnr(x_true, x):.2f}} dB  SSIM {{compute_ssim(x_true, x):.4f}}")

{viz}
```

## Design Your Own

```bash
# Use the 3-agent pipeline (Plan → Judge → Performance)
cd papers/system_design/
python3 main.py --modality {mod_id} --period forward --prompt "your system description"
python3 main.py --modality {mod_id} --period reconstruction --prompt "your algorithm"
```
"""
    return spec


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    modality_dirs = sorted(d for d in AB_DIR.iterdir()
                           if d.is_dir() and d.name != 'shared')
    total = 0
    for mod_dir in modality_dirs:
        solvers_py = mod_dir / 'solvers.py'
        if not solvers_py.exists():
            continue
        mod_id, disp_name, dataset_line, solvers = extract_modality_info(solvers_py)

        if mod_id in PAPER_MAP:
            fwd_name, recon_name = PAPER_MAP[mod_id]
            fwd_path = PAPER_OUTPUTS / fwd_name
            recon_path = PAPER_OUTPUTS / recon_name
            spec = make_spec_from_paper(mod_id, disp_name, fwd_path, recon_path, solvers)
        else:
            spec = make_spec_generic(mod_id, disp_name, dataset_line, solvers)

        (OUT_DIR / f'{mod_id}.md').write_text(spec, encoding='utf-8')
        total += 1

    print(f'Usage 3: generated {total} system design specs in {OUT_DIR}')


if __name__ == '__main__':
    run()
