# X-ray Computed Tomography (CT) — DPS

**GPU**  **PSNR**: ~43.2 dB  *Chung et al., ICML 2023 — 43.2 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('dps', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
