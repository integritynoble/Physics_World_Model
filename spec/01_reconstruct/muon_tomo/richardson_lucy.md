# Muon Tomography — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: muon tracks (N × 6, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/muon_tomo/public/`

```python
from algorithm_base.muon_tomo.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
