# Coherent Diffractive Imaging / Phase Retrieval — HIO + Gradient

**CPU**  **Mismatch**: detector-sample distance error `[-5, +5] mm`
**Input**: diffraction intensities (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/phase_retrieval/public/`

```python
from algorithm_base.phase_retrieval.solvers import run_solver
from pwm_core.mismatch.operators import phase_calibrate_distance

x_wrong = run_solver('traditional_cpu', y)           # no correction
dist_err = phase_calibrate_distance(y)
calib_cfg = {"dist_error": float(dist_err)}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
