# Coded Aperture Snapshot Spectral Imaging (CASSI) — SSR-L + Gradient

**GPU**  **PSNR**: ~34.0 dB  **Mismatch**: dispersion step `[1, 5] px`
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
from pwm_core.mismatch.operators import cassi_calibrate_step

x_wrong = run_solver('ssr_l', y)           # no correction
disp = cassi_calibrate_step(y)
calib_cfg = {"disp_step": float(disp)}
x = run_solver('ssr_l', y, cfg={**calib_cfg, **{'model_key': 'ssr_l'}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
