# Coded Aperture Snapshot Spectral Imaging (CASSI) — BiSRNet

**GPU**  *BiSRNet, 2023*
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
cfg = {'model_key': 'bisrnet'}
x = run_solver('bisrnet', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
