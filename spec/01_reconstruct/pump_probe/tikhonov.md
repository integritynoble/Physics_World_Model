# Pump-Probe Microscopy — Tikhonov Regularization

**CPU**  *Tikhonov, Soviet Math Doklady 1963*
**Input**: transient spectra (T × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pump_probe/public/`

```python
from algorithm_base.pump_probe.solvers import run_solver
cfg = {'iters': 50, 'lam': 0.01, 'step': 0.5}
x = run_solver('tikhonov', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
