# Pump-Probe Microscopy — ReconNet + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: transient spectra (T × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pump_probe/public/`

```python
from algorithm_base.pump_probe.solvers import run_solver


x_wrong = run_solver('dl_reconnet', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_reconnet', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
