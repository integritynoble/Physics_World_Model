# Photometric Stereo — CS-Diffusion + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: images under N lights (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/photometric_stereo/public/`

```python
from algorithm_base.photometric_stereo.solvers import run_solver


x_wrong = run_solver('dl_diffusion', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_diffusion', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
