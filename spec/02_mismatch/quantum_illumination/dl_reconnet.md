# Quantum Illumination — ReconNet + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: coincidence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/quantum_illumination/public/`

```python
from algorithm_base.quantum_illumination.solvers import run_solver


x_wrong = run_solver('dl_reconnet', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_reconnet', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
