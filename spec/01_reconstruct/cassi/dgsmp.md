# Coded Aperture Snapshot Spectral Imaging (CASSI) — DGSMP

**GPU**  **PSNR**: ~32.6 dB  *Huang et al., CVPR 2021 — 32.6 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'model_key': 'dgsmp'}
x = run_solver('dgsmp', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
