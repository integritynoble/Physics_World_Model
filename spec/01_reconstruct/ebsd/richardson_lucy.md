# Electron Backscatter Diffraction (EBSD) — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: Kikuchi pattern (H × W × px × py, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ebsd/public/`

```python
from algorithm_base.ebsd.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
