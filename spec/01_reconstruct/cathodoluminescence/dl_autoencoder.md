# Cathodoluminescence (CL) Imaging — Spec-AE

**GPU**  *Autoencoder spectral unmixing, 2020*
**Input**: spectrum image (H × W × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cathodoluminescence/public/`

```python
from algorithm_base.cathodoluminescence.solvers import run_solver
x = run_solver('dl_autoencoder', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
