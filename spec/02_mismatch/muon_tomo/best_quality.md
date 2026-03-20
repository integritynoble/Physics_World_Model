# Muon Tomography — POCA-DL [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: muon tracks (N × 6, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/muon_tomo/public/`

```python
from algorithm_base.muon_tomo.solvers import run_solver


x_wrong = run_solver('best_quality', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('best_quality', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
