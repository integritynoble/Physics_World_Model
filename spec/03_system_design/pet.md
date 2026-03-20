# Positron Emission Tomography (PET) — System Design

```
[Radiotracer] → [Annihilation] → [Coincidence ring] → y
                                         ↓
                         [OSEM / MAP-EM] → x
                              ↓ attenuation correction
```

**Mismatch**: attenuation map `μ ∈ [0, 0.3] cm⁻¹`
**Input**: sinogram (angles × detectors, float32)  **Algorithms**: 15 — see `spec/pet.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet/public/`

```python
from algorithm_base.pet.solvers import run_solver
from pwm_core.mismatch.operators import pet_calibrate_attenuation
mu_map = pet_calibrate_attenuation(y)
calib_cfg = {"mu_map": mu_map}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
