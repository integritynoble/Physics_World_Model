# XFEL Serial Femtosecond Crystallography (SFX) — TV-ADMM

**CPU**  *Rudin, Osher & Fatemi 1992; Boyd et al. 2010*
**Input**: diffraction patterns (N_shots × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xfel_sfx/public/`

```python
from algorithm_base.xfel_sfx.solvers import run_solver
cfg = {'iters': 20, 'lam': 0.005, 'rho': 1.0}
x = run_solver('tv_admm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
