# X-ray Computed Tomography (CT) — Score-CT

**GPU**  **PSNR**: ~43.0 dB  *Song et al., ICLR 2022 — 43.0 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('score_ct', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
