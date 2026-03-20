# Scanning Transmission Electron Microscopy (STEM) — 3D-Transformer + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: HAADF image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/stem/public/`

```python
from algorithm_base.stem.solvers import run_solver


x_wrong = run_solver('dl_transformer', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_transformer', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
