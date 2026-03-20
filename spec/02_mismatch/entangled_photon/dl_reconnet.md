# Entangled Photon Microscopy — ReconNet + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: coincidence counts (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/entangled_photon/public/`

```python
from algorithm_base.entangled_photon.solvers import run_solver


x_wrong = run_solver('dl_reconnet', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_reconnet', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
