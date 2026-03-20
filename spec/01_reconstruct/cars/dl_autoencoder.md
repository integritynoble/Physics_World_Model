# Coherent Anti-Stokes Raman (CARS) Microscopy — Spec-AE

**GPU**  *Autoencoder spectral unmixing, 2020*
**Input**: CARS spectrum cube (H × W × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cars/public/`

```python
from algorithm_base.cars.solvers import run_solver
x = run_solver('dl_autoencoder', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
