# 4D-STEM Electron Diffraction — ePIE (electron ptychography) + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: diffraction pattern (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_diffraction/public/`

```python
from algorithm_base.electron_diffraction.solvers import run_solver


x_wrong = run_solver('traditional_cpu', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
