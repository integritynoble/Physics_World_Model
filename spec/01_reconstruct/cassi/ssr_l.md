# Coded Aperture Snapshot Spectral Imaging (CASSI) — SSR-L

**GPU**  **PSNR**: ~34.0 dB  *Zhang et al., CVPR 2024 — 34.0 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'model_key': 'ssr_l'}
x = run_solver('ssr_l', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
