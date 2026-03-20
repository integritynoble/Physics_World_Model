# Ptychographic Imaging — Ptychographic Iterative Engine (PIE) + Gradient

**CPU**  **Mismatch**: probe position error `[-3, +3] px`
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
from pwm_core.mismatch.operators import ptycho_calibrate_offset

x_wrong = run_solver('pie', y)           # no correction
pos_err = ptycho_calibrate_offset(y)
calib_cfg = {"pos_error": pos_err}
x = run_solver('pie', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
