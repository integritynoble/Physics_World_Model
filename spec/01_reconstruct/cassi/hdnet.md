# Coded Aperture Snapshot Spectral Imaging (CASSI) — HDNet

**GPU**  **PSNR**: ~34.66 dB  *Hu et al., CVPR 2022 — 34.66 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
x = run_solver('hdnet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
