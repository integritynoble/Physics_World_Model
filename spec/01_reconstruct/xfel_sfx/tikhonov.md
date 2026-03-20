# XFEL Serial Femtosecond Crystallography (SFX) — Tikhonov Regularization

**CPU**  *Tikhonov, Soviet Math Doklady 1963*
**Input**: diffraction patterns (N_shots × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xfel_sfx/public/`

```python
from algorithm_base.xfel_sfx.solvers import run_solver
cfg = {'iters': 50, 'lam': 0.01, 'step': 0.5}
x = run_solver('tikhonov', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
