# Secondary Ion Mass Spectrometry (SIMS) Imaging — Spec-CNN + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: ion images (H × W × m/z, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sims/public/`

```python
from algorithm_base.sims.solvers import run_solver


x_wrong = run_solver('dl_cnn', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_cnn', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
