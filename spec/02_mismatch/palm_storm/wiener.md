# PALM/STORM Single-Molecule Localization — Wiener Deconvolution + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: localisations (N × 4: x,y,σ,I)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/palm_storm/public/`

```python
from algorithm_base.palm_storm.solvers import run_solver


x_wrong = run_solver('wiener', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('wiener', y, cfg={**calib_cfg, **{'reg': 0.01}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
