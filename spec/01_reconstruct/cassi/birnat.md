# Coded Aperture Snapshot Spectral Imaging (CASSI) — BIRNAT

**GPU**  *Cheng et al., ECCV 2022*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'model_key': 'birnat'}
x = run_solver('birnat', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
