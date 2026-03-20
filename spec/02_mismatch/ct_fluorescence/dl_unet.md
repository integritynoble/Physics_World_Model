# CT + Fluorescence (FLIT) — U-Net Recon + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: XRF sinogram (angles × detectors × ch, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct_fluorescence/public/`

```python
from algorithm_base.ct_fluorescence.solvers import run_solver


x_wrong = run_solver('dl_unet', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_unet', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
