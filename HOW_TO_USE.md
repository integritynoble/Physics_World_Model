# How to Use PWM

PWM (Physics World Model) is an open toolkit for computational imaging reconstruction.
It covers **168 imaging modalities** (CT, MRI, ultrasound, lensless, OCT, PET, ...) with
600+ algorithms. You give it a measurement `y`, it gives back a reconstruction `x`.

---

## 1. Setup

### Local (Linux / macOS / Windows)

```bash
git clone https://github.com/integritynoble/Physics_World_Model
cd Physics_World_Model/pwm/public
pip install -e packages/pwm_core
```

### Google Colab

```python
!git clone https://github.com/integritynoble/Physics_World_Model
import sys; sys.path.insert(0, 'Physics_World_Model/pwm/public')
!pip install -e Physics_World_Model/pwm/public/packages/pwm_core -q
```

---

## 2. Get a Spec

A **spec** is a minimal recipe (~15 lines) that tells you exactly what code to run for
your imaging task. There are two ways to get one:

### Without API key — returns the closest preset spec

```bash
python3 spec/autospec.py "CT reconstruction low-dose"
python3 spec/autospec.py "MRI mismatch correction"
python3 spec/autospec.py "lensless imaging system design"
python3 spec/autospec.py "photoacoustic speed-of-sound calibration"
python3 spec/autospec.py list    # show all 168 modalities
```

Example output:
```
Match: X-ray CT
Spec:  spec/ct.md

# X-ray Computed Tomography (CT) — PnP-ADMM (NLM)
**CPU**  **PSNR**: ~39.5 dB
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'iters': 20, 'sigma': 0.05, 'rho': 0.5}
x = run_solver('pnp_admm_nlm', y, cfg=cfg)
```
```

### With API key — LLM auto-designs a custom spec

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 spec/autospec.py "low-dose CT with TV regularization and CoR mismatch correction"
python3 spec/autospec.py "MRI ESPIRiT with coil sensitivity mismatch" --save my_spec.md
```

The LLM reads relevant preset specs as context and generates a spec tailored to your
prompt. Then refine in plain English:

```
You: add data loading from GCS
You: change iterations to 100
You: add visualization
You: save
You: quit
```

---

## 3. Run the Spec

Copy the run button from the spec and execute it:

```python
import numpy as np
from pwm_core.data.loaders import load_benchmark_sample
from algorithm_base.ct.solvers import run_solver
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# Load one benchmark sample
y, x_true = load_benchmark_sample('ct', tier='public', index=0)

# Run (from the spec run button)
cfg = {'iters': 20, 'sigma': 0.05, 'rho': 0.5}
x = run_solver('pnp_admm_nlm', y, cfg=cfg)

# Evaluate
print(f"PSNR: {compute_psnr(x_true, x):.2f} dB")
print(f"SSIM: {compute_ssim(x_true, x):.4f}")
```

---

## 4. The 4 Use Cases

### Use Case 1 — Reconstruct with a specific algorithm

```bash
# Browse all CT algorithms
ls spec/01_reconstruct/ct/

# Read a specific one
cat spec/01_reconstruct/ct/tv_admm.md
```

```python
x = run_solver('tv_admm', y)            # CPU
x = run_solver('ct_fm', y)              # GPU (best quality, ~44.1 dB)
```

### Use Case 2 — Mismatch correction + reconstruct

Handles real-world mismatches (calibration errors, drift, model mismatch):

```bash
cat spec/02_mismatch/ct/pnp_admm_nlm.md
```

```python
from algorithm_base.ct.solvers import run_solver
from pwm_core.mismatch.operators import ct_calibrate_cor

x_wrong = run_solver('pnp_admm_nlm', y)              # no correction
cor_offset = ct_calibrate_cor(y, shift_range=5)       # auto-calibrate
calib_cfg = {"cor_offset": float(cor_offset)}
x = run_solver('pnp_admm_nlm', y, cfg=calib_cfg)     # corrected
```

### Use Case 3 — System design

Full imaging pipeline: source → physics → detector → mismatch → reconstruct:

```bash
cat spec/03_system_design/ct.md
```

### Use Case 4 — Physics simulation

Reproduce benchmark simulations from papers:

```bash
cat spec/04_simulation/09_optics/spec.md   # Fresnel diffraction
```

---

## 5. Compare Algorithms

```python
from algorithm_base.ct.solvers import run_solver, list_solvers

list_solvers()   # prints all 41 CT algorithms with CPU/GPU labels

results = {}
for key in ['traditional_cpu', 'tv_admm', 'pnp_admm_nlm']:   # CPU only
    x = run_solver(key, y)
    results[key] = compute_psnr(x_true, x)
    print(f"{key}: {results[key]:.2f} dB")
```

---

## 6. GPU vs CPU

Specs are labelled **GPU** or **CPU** in the metadata line.

- **CPU** specs run on any machine.
- **GPU** specs raise `RuntimeError` on CPU-only machines — they do not affect CPU specs.

```python
try:
    x = run_solver('ct_fm', y)       # GPU
except RuntimeError:
    x = run_solver('pnp_admm_nlm', y)  # CPU fallback
```

---

## 7. Browse All Modalities

```bash
python3 spec/autospec.py list
# acoustic_emission          Acoustic Emission
# acoustic_microscopy        Acoustic Microscopy
# ct                         X-ray CT
# mri                        MRI
# ...168 total
```

Or browse spec files directly:
```bash
ls spec/               # 168 overview specs + subdirectories
ls spec/01_reconstruct/   # per-algorithm specs
ls spec/02_mismatch/      # mismatch correction specs
ls spec/03_system_design/ # system design specs
ls spec/04_simulation/    # physics simulation specs
```

---

## 8. Benchmark Data

All benchmark data is stored in GCS and downloaded automatically on first use:

```
gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/public/   ← 20 samples, with x_true
gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/dev/      ← 20 samples, blind
gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/hidden/   ← server-only
```

5 real datasets: CT (LoDoPaB-CT), MRI (M4Raw), SD-CASSI (KAIST TSA), CACTI, Hyperspectral (Indian Pines).

---

## Quick Reference

| Task | Command |
|------|---------|
| Get a spec (no API key) | `python3 spec/autospec.py "your query"` |
| Get a spec (with API key) | `ANTHROPIC_API_KEY=... python3 spec/autospec.py "your query"` |
| List all modalities | `python3 spec/autospec.py list` |
| List algorithms for a modality | `ls spec/01_reconstruct/{modality}/` |
| Run an algorithm | `run_solver('key', y)` |
| Evaluate | `compute_psnr(x_true, x)` / `compute_ssim(x_true, x)` |
| GPU fallback | `try: run_solver('gpu_key', y) except RuntimeError: run_solver('cpu_key', y)` |
