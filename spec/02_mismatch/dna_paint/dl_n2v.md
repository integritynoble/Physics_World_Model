# DNA-PAINT Super-Resolution — Noise2Void + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: localisation list (N × 2, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dna_paint/public/`

```python
from algorithm_base.dna_paint.solvers import run_solver


x_wrong = run_solver('dl_n2v', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_n2v', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
