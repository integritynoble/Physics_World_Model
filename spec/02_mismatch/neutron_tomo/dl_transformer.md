# Neutron Radiography / Tomography — TransCT + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_tomo/public/`

```python
from algorithm_base.neutron_tomo.solvers import run_solver


x_wrong = run_solver('dl_transformer', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_transformer', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
