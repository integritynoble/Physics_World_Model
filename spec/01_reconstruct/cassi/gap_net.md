# Coded Aperture Snapshot Spectral Imaging (CASSI) — GAP-Net

**GPU**  **PSNR**: ~29.1 dB  *Meng et al., 2020 — 29.1 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'model_key': 'gap_net'}
x = run_solver('gap_net', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
