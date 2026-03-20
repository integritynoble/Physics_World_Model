# Coded Aperture Snapshot Spectral Imaging (CASSI) — DAUHST-9stg

**GPU**  **PSNR**: ~38.4 dB  *Cai et al., NeurIPS 2022 — 38.4 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'model_key': 'dauhst_9stg'}
x = run_solver('dauhst_9stg', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
