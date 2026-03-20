# Coded Aperture Snapshot Spectral Imaging (CASSI) — MST++

**GPU**  **PSNR**: ~36.0 dB  *Cai et al., CVPRW 2022 — 36.0 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'model_key': 'mst_plus_plus'}
x = run_solver('mst_plus_plus', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
