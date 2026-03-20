# X-ray Computed Tomography (CT) — DOLCE

**GPU**  **PSNR**: ~36.0 dB  *Liu et al., 2023 — 36.0 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('dolce', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
