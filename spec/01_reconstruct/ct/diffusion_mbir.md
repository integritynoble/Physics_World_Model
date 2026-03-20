# X-ray Computed Tomography (CT) — DiffusionMBIR

**GPU**  **PSNR**: ~43.8 dB  *Chung & Ye, NeurIPS 2023 — 43.8 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('diffusion_mbir', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
