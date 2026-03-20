# Coherent Diffractive Imaging / Phase Retrieval — Tikhonov Regularization + Gradient

**CPU**  **Mismatch**: detector-sample distance error `[-5, +5] mm`
**Input**: diffraction intensities (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/phase_retrieval/public/`

```python
from algorithm_base.phase_retrieval.solvers import run_solver
from pwm_core.mismatch.operators import phase_calibrate_distance

x_wrong = run_solver('tikhonov', y)           # no correction
dist_err = phase_calibrate_distance(y)
calib_cfg = {"dist_error": float(dist_err)}
x = run_solver('tikhonov', y, cfg={**calib_cfg, **{'iters': 50, 'lam': 0.01, 'step': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
