# Positron Emission Tomography (PET) — Chambolle-Pock + Gradient

**CPU**  **Mismatch**: attenuation map `μ ∈ [0, 0.3] cm⁻¹`
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet/public/`

```python
from algorithm_base.pet.solvers import run_solver
from pwm_core.mismatch.operators import pet_calibrate_attenuation

x_wrong = run_solver('chambolle_pock', y)           # no correction
mu_map = pet_calibrate_attenuation(y)
calib_cfg = {"mu_map": mu_map}
x = run_solver('chambolle_pock', y, cfg={**calib_cfg, **{'iters': 30, 'lam': 0.005}})        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
