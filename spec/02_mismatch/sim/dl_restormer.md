# Structured Illumination Microscopy (SIM) — Restormer + Gradient

**GPU**  **Mismatch**: illumination phase offset `[0, 2π/3] rad`
**Input**: raw frames (9 × H × W: 3 angles × 3 phases)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sim/public/`

```python
from algorithm_base.sim.solvers import run_solver
from pwm_core.mismatch.operators import sim_calibrate_phase

x_wrong = run_solver('dl_restormer', y)           # no correction
phase = sim_calibrate_phase(y)
calib_cfg = {"phase_offset": float(phase)}
x = run_solver('dl_restormer', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
