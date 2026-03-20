# Coherent Anti-Stokes Raman (CARS) Microscopy — Spec-Transformer

**GPU**  *Transformer for spectroscopy, 2023*
**Input**: CARS spectrum cube (H × W × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cars/public/`

```python
from algorithm_base.cars.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
