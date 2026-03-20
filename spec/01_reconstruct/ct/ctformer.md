# X-ray Computed Tomography (CT) — CTformer

**GPU**  **PSNR**: ~40.8 dB  *Wang et al., IEEE TMI 2023 — 40.8 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('ctformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
