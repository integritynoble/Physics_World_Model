# MR Spectroscopy (MRS) — MedMamba

**GPU**  *SSM for medical imaging, 2026*
**Input**: FID (T, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mrs/public/`

```python
from algorithm_base.mrs.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
