# Digital Breast Tomosynthesis (DBT) — U-Net Recon + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/digital_breast_tomo/public/`

```python
from algorithm_base.digital_breast_tomo.solvers import run_solver


x_wrong = run_solver('dl_unet', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_unet', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
