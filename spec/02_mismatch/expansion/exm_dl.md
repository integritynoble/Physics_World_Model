# Expansion Microscopy (ExM) — EXpansionNet + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: confocal + expansion (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/expansion/public/`

```python
from algorithm_base.expansion.solvers import run_solver


x_wrong = run_solver('exm_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('exm_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
