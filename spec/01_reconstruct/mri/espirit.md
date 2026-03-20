# Magnetic Resonance Imaging (MRI) — ESPIRiT

**CPU**  **PSNR**: ~34.2 dB  *Uecker et al., MRM 2014 — 34.2 dB on fastMRI knee 4x*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('espirit', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
