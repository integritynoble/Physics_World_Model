# Magnetic Resonance Imaging (MRI) — CS-MRI (Wavelet)

**CPU**  **PSNR**: ~33.0 dB  *Lustig et al., MRM 2007 — 33.0 dB on fastMRI knee 4x*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
