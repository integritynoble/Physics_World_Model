# Laser-Induced Breakdown Spectroscopy (LIBS) Imaging — Spec-AE

**GPU**  *Autoencoder spectral unmixing, 2020*
**Input**: emission spectrum (wavelengths, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/libs/public/`

```python
from algorithm_base.libs.solvers import run_solver
x = run_solver('dl_autoencoder', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
