# Radio Aperture Synthesis — TV-ADMM

**CPU**  *Rudin, Osher & Fatemi 1992; Boyd et al. 2010*
**Input**: visibilities (baselines × freq × T, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/radio_astronomy/public/`

```python
from algorithm_base.radio_astronomy.solvers import run_solver
cfg = {'iters': 20, 'lam': 0.005, 'rho': 1.0}
x = run_solver('tv_admm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
