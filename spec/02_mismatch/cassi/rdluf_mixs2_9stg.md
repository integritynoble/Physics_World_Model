# Coded Aperture Snapshot Spectral Imaging (CASSI) — RDLUF-MixS2-9stg + Gradient

**GPU**  **PSNR**: ~39.6 dB  **Mismatch**: dispersion step `[1, 5] px`
**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
from pwm_core.mismatch.operators import cassi_calibrate_step

x_wrong = run_solver('rdluf_mixs2_9stg', y)           # no correction
disp = cassi_calibrate_step(y)
calib_cfg = {"disp_step": float(disp)}
x = run_solver('rdluf_mixs2_9stg', y, cfg={**calib_cfg, **{'model_key': 'rdluf_mixs2_9stg'}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
