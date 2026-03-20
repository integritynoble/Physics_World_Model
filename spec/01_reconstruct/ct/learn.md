# X-ray Computed Tomography (CT) — LEARN

**GPU**  **PSNR**: ~43.1 dB  *Chen et al., IEEE TMI 2018 — 43.1 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('learn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
