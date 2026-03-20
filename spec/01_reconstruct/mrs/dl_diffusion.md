# MR Spectroscopy (MRS) — DiffusionMed

**GPU**  *Diffusion model for medical imaging, 2024*
**Input**: FID (T, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mrs/public/`

```python
from algorithm_base.mrs.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
