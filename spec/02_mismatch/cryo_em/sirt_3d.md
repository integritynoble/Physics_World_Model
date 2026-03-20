# Cryo-EM Single Particle Analysis — SIRT (Simultaneous Iterative) + Gradient

**CPU**  **Mismatch**: defocus value `[0.5, 3.0] μm`
**Input**: particle images (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

```python
from algorithm_base.cryo_em.solvers import run_solver
from pwm_core.mismatch.operators import cryoem_calibrate_defocus

x_wrong = run_solver('sirt_3d', y)           # no correction
defocus = cryoem_calibrate_defocus(y)
calib_cfg = {"defocus": float(defocus)}
x = run_solver('sirt_3d', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
