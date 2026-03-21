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

A **spec** is a minimal recipe (~15 lines) that tells you exactly what code to run.

### Without API key — returns the closest preset spec

```bash
python3 spec/autospec.py "CT reconstruction low-dose"
python3 spec/autospec.py "MRI mismatch correction"
python3 spec/autospec.py "lensless imaging system design"
python3 spec/autospec.py list    # show all 168 modalities
```

### With API key — LLM auto-designs a custom spec

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 spec/autospec.py "low-dose CT with TV regularization and CoR mismatch"
python3 spec/autospec.py "MRI ESPIRiT with coil sensitivity mismatch" --save my_spec.md
```

Refine in plain English after the spec is generated:

```
You: add data loading from GCS
You: change iterations to 100
You: save
You: quit
```

---

## 3. Run the Spec

### Step 1 — Load benchmark data

```python
import sys
sys.path.insert(0, 'Physics_World_Model/pwm/public')   # adjust path if needed

# Option A: download from GCS (requires gsutil or google-cloud-storage)
from scripts.gcs_dataset_helper import ensure_challenge_dataset
import h5py, numpy as np

h5_path = ensure_challenge_dataset('ct', 'public')  # downloads to /tmp/pwm_challenge_cache/
with h5py.File(h5_path, 'r') as f:
    y      = f['y'][0].astype('float32')     # sinogram, shape (angles, detectors)
    x_true = f['x_true'][0].astype('float32')

# Option B: use your own measurement
y = np.load('my_sinogram.npy').astype('float32')
```

### Step 2 — Run the solver (copy from spec run button)

```python
from algorithm_base.ct.solvers import run_solver

cfg = {'iters': 20, 'sigma': 0.05, 'rho': 0.5}
x = run_solver('pnp_admm_nlm', y, cfg=cfg)
print('reconstruction shape:', x.shape)
```

### Step 3 — Evaluate

```python
from pwm_core.analysis.metrics import psnr, mse
import numpy as np

print(f"PSNR: {psnr(x_true, x):.2f} dB")
print(f"MSE:  {mse(x_true, x):.6f}")
```

### Step 4 — Visualize

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(y,      cmap='gray'); axes[0].set_title('Measurement (sinogram)')
axes[1].imshow(x,      cmap='gray'); axes[1].set_title('Reconstruction')
axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout(); plt.savefig('ct_result.png'); plt.show()
```

---

## 4. The 4 Use Cases

### Use Case 1 — Reconstruct with a specific algorithm

```bash
ls spec/01_reconstruct/ct/      # 41 CT algorithms
cat spec/01_reconstruct/ct/tv_admm.md
```

```python
x = run_solver('tv_admm', y)             # CPU
x = run_solver('pnp_admm_nlm', y)        # CPU, ~39.5 dB
x = run_solver('ct_fm', y)               # GPU, ~44.1 dB
```

### Use Case 2 — Mismatch correction + reconstruct

```bash
cat spec/02_mismatch/ct/pnp_admm_nlm.md
```

```python
from algorithm_base.ct.solvers import run_solver
from pwm_core.mismatch.operators import ct_calibrate_cor

x_wrong = run_solver('pnp_admm_nlm', y)              # no correction
cor_offset = ct_calibrate_cor(y, shift_range=5)       # auto-calibrate CoR
calib_cfg = {"cor_offset": float(cor_offset)}
x = run_solver('pnp_admm_nlm', y, cfg=calib_cfg)     # corrected

print(f"Before: {psnr(x_true, x_wrong):.2f} dB")
print(f"After:  {psnr(x_true, x):.2f} dB")
```

### Use Case 3 — System design

```bash
cat spec/03_system_design/ct.md    # shows full pipeline DAG + run button
```

### Use Case 4 — Physics simulation

```bash
cat spec/04_simulation/09_optics/spec.md   # Fresnel diffraction
# full spec at: papers/universal_simulation/benchmark/09_optics/spec.md
```

---

## 5. Compare Algorithms

```python
from algorithm_base.ct.solvers import run_solver, list_solvers
from pwm_core.analysis.metrics import psnr

# See all available algorithms
for key, info in list_solvers():
    gpu = 'GPU' if info.get('gpu') else 'CPU'
    print(f"{key:<25} {gpu}  {info.get('name','')}")

# Compare CPU algorithms
for key in ['traditional_cpu', 'tv_admm', 'pnp_admm_nlm']:
    x = run_solver(key, y)
    print(f"{key}: {psnr(x_true, x):.2f} dB")
```

---

## 6. GPU vs CPU

Specs are labelled **GPU** or **CPU** in the metadata line.

- **CPU** specs run on any machine — no GPU needed.
- **GPU** specs raise `RuntimeError` on CPU-only machines — this does not affect CPU specs.

```python
try:
    x = run_solver('ct_fm', y)              # GPU — best quality
except RuntimeError:
    x = run_solver('pnp_admm_nlm', y)       # CPU fallback
```

---

## 7. Browse All Modalities

```bash
python3 spec/autospec.py list
# ct                         X-ray CT
# mri                        MRI
# ultrasound                 Ultrasound
# ...168 total

ls spec/               # 168 overview specs + 4 use-case subdirectories
ls spec/01_reconstruct/ct/     # all 41 CT algorithm specs
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Get a spec (no API key) | `python3 spec/autospec.py "your query"` |
| Get a spec (with API key) | `ANTHROPIC_API_KEY=... python3 spec/autospec.py "query"` |
| List all modalities | `python3 spec/autospec.py list` |
| List algorithms | `ls spec/01_reconstruct/{modality}/` |
| Download benchmark data | `ensure_challenge_dataset('ct', 'public')` |
| Run an algorithm | `run_solver('key', y)` |
| Evaluate | `psnr(x_true, x)` / `mse(x_true, x)` |
| GPU fallback | `try: run_solver('gpu_key', y) except RuntimeError: run_solver('cpu_key', y)` |
