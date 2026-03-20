# Coded Aperture Snapshot Spectral Imaging (CASSI) — MiJUN-5stg

**GPU**  **PSNR**: ~40.9 dB  *Meng et al., AAAI 2025 — 40.9 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'model_key': 'mijun_5stg'}
x = run_solver('mijun_5stg', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
