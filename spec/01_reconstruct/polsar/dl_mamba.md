# Polarimetric SAR (PolSAR) — RS-Mamba

**GPU**  *SSM for remote sensing, 2026*
**Input**: scattering matrix (H × W × 4, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/polsar/public/`

```python
from algorithm_base.polsar.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
