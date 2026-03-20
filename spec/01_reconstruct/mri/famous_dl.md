# Magnetic Resonance Imaging (MRI) — MoDL

**GPU**  **PSNR**: ~36.0 dB  *Aggarwal et al., IEEE TMI 2019 — 36.0 dB on fastMRI knee 4x*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
