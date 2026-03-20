# Ptychographic Imaging — System Design

```
[Focused probe] → [Sample] → [Diffraction] → y
                                     ↓
                  [ePIE / ADMM-ptycho] → x
                       ↓ position calibration
```

**Mismatch**: probe position error `[-3, +3] px`
**Input**: diffraction patterns (N_pos × H × W, float32)  **Algorithms**: 17 — see `spec/ptychography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
from pwm_core.mismatch.operators import ptycho_calibrate_offset
pos_err = ptycho_calibrate_offset(y)
calib_cfg = {"pos_error": pos_err}
x = run_solver('error_reduction', y, cfg=calib_cfg)
```
