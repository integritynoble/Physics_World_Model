# Industrial X-ray CT — U-Net Recon + Gradient

**GPU**  **Mismatch**: center-of-rotation offset `[-10, +10] px`
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/industrial_ct/public/`

```python
from algorithm_base.industrial_ct.solvers import run_solver
from pwm_core.mismatch.operators import ct_calibrate_cor

x_wrong = run_solver('dl_unet', y)           # no correction
cor_offset = ct_calibrate_cor(y, shift_range=10)
calib_cfg = {"cor_offset": float(cor_offset)}
x = run_solver('dl_unet', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
