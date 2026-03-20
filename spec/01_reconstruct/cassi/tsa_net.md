# Coded Aperture Snapshot Spectral Imaging (CASSI) — TSA-Net

**GPU**  **PSNR**: ~31.5 dB  *Meng et al., ECCV 2020 — 31.5 dB on KAIST*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'model_key': 'tsa_net'}
x = run_solver('tsa_net', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
