# Coded Aperture Snapshot Spectral Imaging (CASSI) — TwIST + Gradient

**CPU**  **PSNR**: ~23.1 dB  **Mismatch**: dispersion step `[1, 5] px`
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
from pwm_core.mismatch.operators import cassi_calibrate_step

x_wrong = run_solver('twist', y)           # no correction
disp = cassi_calibrate_step(y)
calib_cfg = {"disp_step": float(disp)}
x = run_solver('twist', y, cfg={**calib_cfg, **{'iters': 100, 'lam': 0.01, 'tv_iter': 5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
