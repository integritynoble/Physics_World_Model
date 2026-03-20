# MR Spectroscopy (MRS) — SwinIR-Med

**GPU**  *Liang et al., ICCV 2021*
**Input**: FID (T, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mrs/public/`

```python
from algorithm_base.mrs.solvers import run_solver
x = run_solver('dl_swinir', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
