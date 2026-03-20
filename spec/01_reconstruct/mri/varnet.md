# Magnetic Resonance Imaging (MRI) — E2E-VarNet

**GPU**  **PSNR**: ~40.5 dB  *Sriram et al., MICCAI 2020 — 40.5 dB on fastMRI knee 4x*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('varnet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
