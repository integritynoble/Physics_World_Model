# Diffusion MRI (DTI) — SENSE (WLS tensor fit)

**CPU**
**Input**: DWI (N_dirs × H × W × D, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/diffusion_mri/public/`

```python
from algorithm_base.diffusion_mri.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
