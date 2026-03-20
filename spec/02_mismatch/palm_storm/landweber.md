# PALM/STORM Single-Molecule Localization — Landweber Iteration + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: localisations (N × 4: x,y,σ,I)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/palm_storm/public/`

```python
from algorithm_base.palm_storm.solvers import run_solver


x_wrong = run_solver('landweber', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('landweber', y, cfg={**calib_cfg, **{'iters': 50, 'step': 0.5}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
