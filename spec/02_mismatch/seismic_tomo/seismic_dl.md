# Seismic Tomography — SeisInversion-Net [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: travel times (src-recv, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/seismic_tomo/public/`

```python
from algorithm_base.seismic_tomo.solvers import run_solver


x_wrong = run_solver('seismic_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('seismic_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
